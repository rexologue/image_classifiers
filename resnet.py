from typing import Literal 

import torch
import torch.nn.functional as F
from torch.nn import (Linear, Conv2d, BatchNorm1d, BatchNorm2d, 
                      ReLU, Sigmoid, SiLU, 
                      AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, 
                      Sequential, Module) 

from utils import Flatten, Normalize


class SEModule(Module):
    def __init__(self, channels, reduction):
        """
        Механизм Squeeze-and-Excitation (SE) представляет собой модуль, для адаптивной перекалибровки каналов признаков. Его основная идея – динамически выделять 
        более информативные каналы и подавлять менее важные. Принцип работы SE-модуля:

        1. Squeeze (Сжатие)
        На этом этапе пространственная информация сводится к вектору, отражающему глобальные характеристики по каждому каналу. 
        Для этого применяется Global Average Pooling.  
        - Цель: Собрать статистическую сводку (агрегировать) по всему пространству для каждого канала.
        - Результат: Для входного тензора размером H x W x C получается вектор размерности 1 x 1 x C, где каждый элемент описывает важность соответствующего канала в целом изображении.

        2. Excitation (Возбуждение)
        После сжатия производится «возбуждение» – восстановление и калибровка информации для каждого канала.  
        - Процесс: Вектор из шага Squeeze пропускается через небольшую нейронную сеть, обычно состоящую из двух полносвязных слоев с нелинейностями.  
        - Первый слой: Уменьшает размерность (с помощью коэффициента редукции, например, reduction), что позволяет моделировать взаимосвязи между каналами.
        - Второй слой: Восстанавливает исходное число каналов.  
        - Функция активации: Часто используется ReLU после первого слоя и сигмоида после второго, что позволяет получить веса в диапазоне [0, 1] для каждого канала.
        - Результат: Получается вектор с весовыми коэффициентами для каждого канала, отражающий его относительную важность.

        3. Recalibration (Перекалибровка)
        Последним этапом является масштабирование исходных признаков с учетом вычисленных коэффициентов.  
        - Операция: Каждый канал входного тензора умножается на соответствующий скаляр из вектора возбуждения.  
        - Эффект: Это позволяет сети «перекалибровать» активацию каналов – усилить те, которые важны, и ослабить менее значимые, 
                  тем самым улучшая представление признаков для последующих слоев.

        Итоговая идея: Модуль SE позволяет сети динамически корректировать вклад каждого канала в зависимости от глобального контекста.
        """
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1) # Приводим все feature maps к вектору размерности 

        # Здесь свертка выступает в качетсве линейного слоя, сжимающего вход. Так как изначально у нас были матрицы
        # feature maps размерности HxW, то после пулинга мы получим матрицы размерности 1x1. Чтобы не решейпить
        # тензоры, применяем такой подход.
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        # Инициализируем слой
        torch.nn.init.xavier_uniform_(self.fc1.weight.data)

        # Второй слой, восстанавливающий исходную размерность
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        # Активации
        self.relu = ReLU(inplace=True)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # Squeeze
        out = self.avg_pool(x)

        # Exication
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        # Recolibration
        return out * x


class ImprovedBottleneck(Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 downsample=True,
                 use_se=True,
                 reduction_ratio=16):
        """
        Улучшенный Bottleneck Block для IR50 (на базе ResNet-v2 + SE-блоки).

        Компоненты блока:
        
        1. Shortcut Path 
        - Если downsample=True: уменьшение разрешения через AvgPool2d + 1x1 свёртка (ResNet-D стиль).
        - Если in_channels != out_channels: выравнивание каналов 1x1 свёрткой.
        - Сохраняет градиенты за счёт остаточных связей.

        2. Residual Path
        - Архитектура bottleneck: 1x1 → 3x3 → 1x1 свёртки.
        - Порядок слоёв: BatchNorm → SiLU → Conv (предварительная активация).
        - Опционально: SE-блок для перевзвешивания каналов.

        Параметры:
            in_channels (int): Число входных каналов.
            out_channels (int): Число выходных каналов.
            downsample (bool): Уменьшение spatial-размера в 2 раза (по умолчанию True).
            use_se (bool): Включение SE-блока (по умолчанию True).
            reduction_ratio (int): Степень сжатия в SE-блоке (по умолчанию 16).
        """
        super(ImprovedBottleneck, self).__init__()
        
        #################
        # Shortcut Path #
        #################
        self.shortcut_layer = Sequential()

        # Даунсэмплинг через AvgPool (если требуется)
        if downsample:
            self.shortcut_layer.add_module("pool", AvgPool2d(kernel_size=2, stride=2))

        # Выравнивание каналов (если требуется)
        if in_channels != out_channels:
            self.shortcut_layer.add_module("conv", Conv2d(
                in_channels, out_channels, 
                kernel_size=1, stride=1, 
                bias=False
            ))
            self.shortcut_layer.add_module("bn", BatchNorm2d(out_channels))

        #################
        # Residual Path #
        #################
        bottleneck_channels = out_channels // 4
        stride = 2 if downsample else 1

        self.res_layer = Sequential(
            # Фаза 1: Сжатие каналов
            BatchNorm2d(in_channels),
            SiLU(inplace=True),
            Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False),
            
            # Фаза 2: Пространственная обработка
            BatchNorm2d(bottleneck_channels),
            SiLU(inplace=True),
            Conv2d(bottleneck_channels, bottleneck_channels, 
                   kernel_size=3, stride=stride, padding=1, bias=False),
            
            # Фаза 3: Восстановление каналов
            BatchNorm2d(bottleneck_channels),
            SiLU(inplace=True),
            Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(out_channels)
        )

        if use_se:
            self.res_layer.add_module("se", SEModule(out_channels, reduction_ratio))

    def forward(self, x):
        """Прямой проход: F(x) + Shortcut(x)"""
        return self.res_layer(x) + self.shortcut_layer(x)


class ClassicBottleneck(Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 downsample=True):
        """
        Классический Bottleneck Block для ResNet (v1, post-activation).

        Компоненты блока:
        
        1. Shortcut Path:
           - Если downsample=True, уменьшаем spatial-размер (stride=2) 
             в 1x1-свёртке.
           - Если in_channels != out_channels, выравниваем число каналов 
             тем же 1x1 Conv.

        2. Residual Path:
           - Архитектура bottleneck: 1x1 → 3x3 → 1x1 свёртки 
             (каналы "сжимаются" и затем "расширяются" в 4 раза).
           - Порядок слоёв: Conv → BN → ReLU (post-activation).
           - Финальный ReLU выполняется после сложения с shortcut.
        
        Параметры:
            in_channels (int): Число входных каналов.
            out_channels (int): Число выходных каналов.
            downsample (bool): Флаг уменьшения spatial-размера в 2 раза (stride=2).
        """
        super(ClassicBottleneck, self).__init__()

        # Вычисляем stride для блока и коэффициент сжатия
        stride = 2 if downsample else 1
        bottleneck_channels = out_channels // 4

        #################
        # Shortcut Path #
        #################
        self.shortcut_layer = Sequential()

        # Если надо уменьшить размер (stride=2) или выровнять каналы:
        if downsample or (in_channels != out_channels):
            self.shortcut_layer.add_module(
                "conv", 
                Conv2d(in_channels, out_channels, 
                       kernel_size=1, stride=stride, bias=False)
            )
            self.shortcut_layer.add_module(
                "bn", 
                BatchNorm2d(out_channels)
            )

        #################
        # Residual Path #
        #################

        bottleneck_channels = out_channels // 4
        stride = 2 if downsample else 1

        self.res_layer = Sequential(
            # Фаза 1: Сжатие каналов
            Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(bottleneck_channels),
            ReLU(inplace=True),
            
            # Фаза 2: Пространственная обработка
            Conv2d(bottleneck_channels, bottleneck_channels, 
                   kernel_size=3, stride=stride, padding=1, bias=False),
            BatchNorm2d(bottleneck_channels),
            ReLU(inplace=True),
            
            # Фаза 3: Восстановление каналов
            Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(out_channels)
        )

    def forward(self, x):
        """Прямой проход (пост-активация): F(x) + Shortcut(x), затем ReLU."""
        return F.relu(self.res_layer(x) + self.shortcut_layer(x))


class BottleneckBlock(Module):
    def __init__(self, 
                arch_type: Literal['classic', 'improved'],
                in_channels: int, 
                out_channels: int, 
                downsample=True, 
                use_se=True, 
                reduction_ratio=16):

        super(BottleneckBlock, self).__init__()

        if arch_type == 'classic':
            self.block = ClassicBottleneck(in_channels, out_channels, downsample)
        
        if arch_type == 'improved':
            self.block = ImprovedBottleneck(in_channels, out_channels, downsample, use_se, reduction_ratio)

    def forward(self, x):
        return self.block(x)


class ResidualStemBlock(Module):
    def __init__(self, arch_type: Literal['classic', 'improved']):
        super(ResidualStemBlock, self).__init__()

        if arch_type == 'classic':
            self.stem = Sequential(
                Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                BatchNorm2d(64),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=3, stride=2, padding=1) 
            )
        
        if arch_type == 'improved':
            self.stem = Sequential(
                Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                BatchNorm2d(64),
                SiLU(inplace=True),

                Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
                BatchNorm2d(64),
                SiLU(inplace=True),
            )

    def forward(self, x):
        return self.stem(x)


class ResNet(Module):
    def __init__(self, arch_type: Literal['classic', 'improved'], blocks_amount: Literal[50, 101, 152]):
        super(ResNet, self).__init__()
       
        self.arch_type = arch_type
        self.blocks_amount = blocks_amount

        if blocks_amount == 50:
            blocks_per_stage = [3, 4, 6, 3]
        elif blocks_amount == 101:
            blocks_per_stage = [3, 4, 23, 3]
        elif blocks_amount == 152:
            blocks_per_stage = [3, 8, 36, 3]

        channels_per_stage = [256, 512, 1024, 2048, 64]

        # Предобработка
        self.stem = ResidualStemBlock(arch_type)

        # Основное тело модели
        self.stages = Sequential()

        for stage_idx in range(4):
            current_blocks_amount = blocks_per_stage[stage_idx]
            previous_channels_amount = channels_per_stage[stage_idx - 1]
            current_channels_amount = channels_per_stage[stage_idx]

            stage = self._make_stage(
                num_blocks=current_blocks_amount,
                in_channels=previous_channels_amount,
                out_channels=current_channels_amount
            )

            self.stages.add_module(f"stage_{stage_idx}", stage)

        # Финальный слой пулинга
        self.final_pool = Sequential(
            AdaptiveAvgPool2d(1),
            Flatten()
        )


    def _make_stage(self, 
                    num_blocks, 
                    in_channels, 
                    out_channels):
        """
        Создает этап сети из нескольких BottleneckIR блоков.
        
        Аргументы:
            num_blocks (int): Количество блоков в этапе
            in_channels (int): Количество входных каналов для первого блока
            out_channels (int): Количество выходных каналов для всех блоков
        """
        stage = Sequential()

        use_se = True if self.arch_type == 'improved' else False
        reduction_ratio = 16
        
        # Первый блок уменьшает размерность
        stage.add_module('block_0', BottleneckBlock(
            arch_type=self.arch_type,
            in_channels=in_channels,
            out_channels=out_channels,
            downsample=True,
            use_se=use_se,
            reduction_ratio=reduction_ratio
        ))
        
        # Последующие блоки работают без изменения размера
        for block_idx in range(1, num_blocks):
            stage.add_module(f'block_{block_idx}', BottleneckBlock(
                arch_type=self.arch_type,
                in_channels=out_channels,
                out_channels=out_channels,
                downsample=False,
                use_se=use_se,
                reduction_ratio=reduction_ratio
            ))
            
        return stage


    def forward(self, x):
        out = self.stem(x)
        out = self.stages(out)
        out = self.final_pool(out)

        return out

        
class FaceEmbedder(Module):
    def __init__(self, 
                 blocks_amount: Literal[50, 101, 152],
                 embedding_size=512):
        """
        Полная сеть для получения векторного представления лица.
        
        Аргументы:
            embedding_size (int): Размер выходного вектора-эмбеддинга
            backbone_config (dict): Конфигурация базовой сети
        """
        super(FaceEmbedder, self).__init__()
        
        # Базовая сеть для извлечения признаков
        self.resnet = ResNet('improved', blocks_amount)
        
        # Финальные слои для преобразования в эмбеддинг
        self.embedder = Sequential(
            Linear(2048, embedding_size),  # Линейный слой
            BatchNorm1d(embedding_size),   # Нормализация
            SiLU(inplace=True),            # Активация
            Normalize('l2')                # Нормализация
        )
        
    def forward(self, x):
        # Извлечение признаков базовой сетью
        features = self.resnet(x)
        # Преобразование в эмбеддинг
        embedding = self.embedder(features)
        
        return embedding


class ResNetClassifier(Module):
    def __init__(self, arch_type: Literal['classic', 'improved'], blocks_amount: Literal[50, 101, 152], num_classes: int):
        super(ResNetClassifier, self).__init__()

        self.resnet = ResNet(arch_type, blocks_amount)
        self.classifier = Linear(2048, num_classes)

    def forward(self, x):
        return self.classifier(self.resnet(x))

