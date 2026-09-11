import torch
import torch.nn as nn



class Unsqueeze(nn.Module):
    def __init__(self, multiChannel=False, dataDim=1):
        super().__init__()
        self.dataDim = dataDim
        self.multiChannel = multiChannel

    def forward(self, x):
        if self.multiChannel:
            if x.ndim == self.dataDim:
                x = torch.unsqueeze(x, 0)
        else:
            if x.ndim == self.dataDim:
                x = torch.unsqueeze(x, 0)
            if x.ndim == self.dataDim + 1:
                x = torch.unsqueeze(x, 1)
        return x


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim == -1:
            return torch.squeeze(x)
        return torch.squeeze(x, self.dim)


def ActUnit(actUnit="ReLU", inplace=False):
    name = actUnit.lower()

    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "relu6":
        return nn.ReLU6(inplace=inplace)
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=inplace)
    if name == "gelu":
        return nn.GELU()
    if name == "selu":
        return nn.SELU(inplace=inplace)
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    if name == "swish":
        return nn.Hardswish(inplace=inplace)
    if name == "elu":
        return nn.ELU(inplace=inplace)
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "softsign":
        return nn.Softsign()
    if name == "softplus":
        return nn.Softplus()
    if name == "identity":
        return nn.Identity()

    raise ValueError(f"Unsupported activation: {actUnit}")


def BatchActUnit(xSize, actUnit="ReLU", batchNorm=False):
    mainProc = []

    if batchNorm:
        mainProc.append(nn.BatchNorm1d(xSize))

    if actUnit:
        mainProc.append(ActUnit(actUnit))

    if not mainProc:
        mainProc.append(nn.Identity())

    return nn.Sequential(*mainProc)



def ConvOnly(inCh, outCh, convType="auto", kernel_size=-1):
    if convType == "auto":
        if outCh == 0:
            convType = "bypass"
        elif inCh > outCh:
            convType = "up"
        elif inCh < outCh:
            convType = "down"
        else:
            convType = "same"

    if convType == "bypass":
        return nn.Identity()

    if convType == "same1x":
        return nn.Conv1d(
            inCh, outCh,
            kernel_size=1,
            stride=1,
            padding=0
        )

    if convType == "same":
        if kernel_size == -1:
            kernel_size = 3

        return nn.Conv1d(
            inCh, outCh,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )

    if convType == "down":
        if kernel_size == -1:
            kernel_size = 4

        return nn.Conv1d(
            inCh, outCh,
            kernel_size=kernel_size,
            stride=2,
            padding=(kernel_size - 1) // 2
        )

    if convType == "up":
        if kernel_size == -1:
            kernel_size = 4

        return nn.ConvTranspose1d(
            inCh, outCh,
            kernel_size=kernel_size,
            stride=2,
            padding=(kernel_size - 1) // 2,
            output_padding=kernel_size % 2
        )

    raise ValueError(f"Unsupported convType: {convType}")


def ConvCell(
    xSize,
    ySize,
    convType="auto",
    kernel_size=-1,
    preAct=True,
    actUnit="ReLU",
    batchNorm=False
):
    mainProc = []

    if preAct:
        mainProc.append(
            BatchActUnit(xSize, actUnit, batchNorm)
        )

    mainProc.append(
        ConvOnly(xSize, ySize, convType, kernel_size)
    )

    if not preAct:
        mainProc.append(
            BatchActUnit(ySize, actUnit, batchNorm)
        )

    return nn.Sequential(*mainProc)


def ConvCellBeforeAdd(
    xSize,
    ySize,
    convType="auto",
    kernel_size=-1,
    preAct=True,
    actUnit="ReLU",
    batchNorm=False
):
    mainProc = []

    if preAct:
        mainProc.append(
            BatchActUnit(xSize, actUnit, batchNorm)
        )

    mainProc.append(
        ConvOnly(xSize, ySize, convType, kernel_size)
    )

    if not preAct and batchNorm:
        mainProc.append(
            nn.BatchNorm1d(ySize)
        )

    return nn.Sequential(*mainProc)



class SimpleCoreCell(nn.Module):
    def __init__(
        self,
        coreCh,
        useInCh=False,
        highCh=0,
        lowCh=0,
        resNet=False,
        preAct=True,
        actUnit="ReLU",
        batchNorm=False
    ):
        super().__init__()

        if resNet is False:
            preAct = False

        self.preAct = preAct
        self.useInCh = useInCh

        if not (useInCh or lowCh or highCh):
            raise ValueError(
                "At least one of useInCh, lowCh, or highCh must be enabled."
            )

        if highCh:
            if highCh == -1:
                highCh = coreCh

            self.highProcess = ConvCellBeforeAdd(
                highCh,
                coreCh,
                "down",
                -1,
                preAct,
                actUnit,
                batchNorm
            )

        if lowCh:
            if lowCh == -1:
                lowCh = coreCh

            self.lowProcess = ConvCellBeforeAdd(
                lowCh,
                coreCh,
                "up",
                -1,
                preAct,
                actUnit,
                batchNorm
            )

        self.actUnit = ActUnit(actUnit)

        self.coreProcessA = ConvCell(
            coreCh,
            coreCh,
            "same",
            -1,
            preAct,
            actUnit,
            batchNorm
        )

        self.coreProcessB = ConvCellBeforeAdd(
            coreCh,
            coreCh,
            "same",
            -1,
            preAct,
            actUnit,
            batchNorm
        )

        self.lowCh = lowCh
        self.highCh = highCh
        self.resNet = resNet

    def forward(self, x=None, y=None, z=None):
        # x: same-resolution input
        # y: higher-resolution input, downsampled before addition
        # z: lower-resolution input, upsampled before addition

        xA = x

        if y is not None:
            yA = self.highProcess(y)
            xA = xA + yA if xA is not None else yA

        if z is not None:
            zA = self.lowProcess(z)
            xA = xA + zA if xA is not None else zA

        if not self.preAct:
            xA = self.actUnit(xA)

        outPut = self.coreProcessB(
            self.coreProcessA(xA)
        )

        if self.resNet:
            outPut = outPut + xA

        if not self.preAct:
            outPut = self.actUnit(outPut)

        return outPut



def InitialCNN(
    layerShape,
    preAct=True,
    actUnit="ReLU",
    batchNorm=False
):
    inCh, outCh, convType = layerShape

    if inCh <= 0:
        raise ValueError("inCh must be greater than 0.")

    mainProc = [
        Unsqueeze(
            multiChannel=(inCh > 1),
            dataDim=1
        ),
        ConvOnly(
            inCh,
            outCh,
            convType
        )
    ]

    if not preAct:
        mainProc.append(
            BatchActUnit(
                outCh,
                actUnit,
                batchNorm
            )
        )

    return nn.Sequential(*mainProc)


def FinalCNN(
    inCh,
    outShape=(1, 512),
    convType="same",
    outCounts=0,
    preAct=True,
    actUnit="ReLU",
    batchNorm=False
):
    if outCounts != 0:
        raise ValueError(
            "This inference implementation supports regression only."
        )

    if tuple(outShape) != (1, 512):
        raise ValueError(
            "This DU-GridNet implementation expects outShape=(1, 512)."
        )

    outCh = 1
    mainProc = []

    if outCh != inCh:
        if preAct:
            mainProc.append(
                BatchActUnit(
                    inCh,
                    actUnit,
                    batchNorm
                )
            )

        mainProc.append(
            ConvOnly(
                inCh,
                outCh,
                convType
            )
        )

    mainProc.append(
        Squeeze()
    )

    return nn.Sequential(*mainProc)



class GridDownCell(nn.Module):
    def __init__(
        self,
        ioShape,
        useInput=False,
        outShape=-1,
        outCounts=0,
        repeatCounts=1,
        simpleCore=True,
        resNet=False,
        preAct=True,
        actUnit="ReLU",
        batchNorm=False
    ):
        super().__init__()

        if not simpleCore:
            raise ValueError(
                "This DU-GridNet implementation uses simpleCore=True."
            )

        self.layerCounts = len(ioShape)
        self.useInput = useInput
        self.outShape = outShape

        mainLayer = []

        for _ in range(repeatCounts):
            downLayer = [
                SimpleCoreCell(
                    ioShape[0],
                    True,
                    0,
                    0,
                    resNet,
                    preAct,
                    actUnit,
                    batchNorm
                )
            ]

            for i in range(1, self.layerCounts):
                downLayer.append(
                    SimpleCoreCell(
                        ioShape[i],
                        True,
                        ioShape[i - 1],
                        0,
                        resNet,
                        preAct,
                        actUnit,
                        batchNorm
                    )
                )

            mainLayer.append(
                nn.ModuleList(downLayer)
            )

        self.mainLayer = nn.ModuleList(mainLayer)

        if outShape is None:
            return

        if outShape == -1:
            outShape = (1, 512)

        if outCounts < 1:
            raise ValueError(
                "outCounts must be >= 1 when GridDownCell produces final output."
            )

        self.outLayer = FinalCNN(
            ioShape[-1],
            outShape,
            "same",
            outCounts,
            preAct,
            actUnit,
            batchNorm
        )

    def forward(self, inX):
        outX = (
            [inX] + [None] * (self.layerCounts - 1)
            if self.useInput
            else inX
        )

        for vLayer in self.mainLayer:
            y = None

            for k, iLayer in enumerate(vLayer):
                outX[k] = y = iLayer(
                    outX[k],
                    y
                )

        if self.outShape is None:
            return outX

        return self.outLayer(y)


class GridUpCell(nn.Module):
    def __init__(
        self,
        ioShape,
        useInput=False,
        outShape=None,
        repeatCounts=1,
        simpleCore=True,
        resNet=False,
        preAct=True,
        actUnit="ReLU",
        batchNorm=False
    ):
        super().__init__()

        if not simpleCore:
            raise ValueError(
                "This DU-GridNet implementation uses simpleCore=True."
            )

        self.layerCounts = len(ioShape)
        self.useInput = useInput
        self.outShape = outShape

        mainLayer = []

        for _ in range(repeatCounts):
            upLayer = [
                SimpleCoreCell(
                    ioShape[0],
                    True,
                    0,
                    0,
                    resNet,
                    preAct,
                    actUnit,
                    batchNorm
                )
            ]

            for i in range(1, self.layerCounts):
                upLayer.append(
                    SimpleCoreCell(
                        ioShape[i],
                        True,
                        0,
                        ioShape[i - 1],
                        resNet,
                        preAct,
                        actUnit,
                        batchNorm
                    )
                )

            mainLayer.append(
                nn.ModuleList(upLayer)
            )

        self.mainLayer = nn.ModuleList(mainLayer)

        if outShape is None:
            return

        if outShape == -1:
            outShape = (1, 512)

        self.outLayer = FinalCNN(
            ioShape[-1],
            outShape,
            "same",
            0,
            preAct,
            actUnit,
            batchNorm
        )

    def forward(self, x):
        inX = (
            [x] + [None] * (self.layerCounts - 1)
            if self.useInput
            else x[::-1]
        )

        for vLayer in self.mainLayer:
            z = None

            for k, iLayer in enumerate(vLayer):
                inX[k] = z = iLayer(
                    inX[k],
                    None,
                    z
                )

        if self.outShape is not None:
            return self.outLayer(z)

        return inX[::-1]



class GridCell(nn.Module):
    def __init__(
        self,
        ioShape,
        useInput=True,
        outShape=(1, 512),
        outCounts=0,
        repeatCounts=1,
        altCounts=4,
        simpleCore=True,
        resNet=False,
        preAct=True,
        actUnit="ReLU",
        batchNorm=False
    ):
        super().__init__()

        if altCounts < 1:
            raise ValueError("altCounts must be >= 1.")

        mainLayer = []
        noInput = False

        while altCounts:
            if altCounts == 2:
                mainLayer.append(
                    GridDownCell(
                        ioShape,
                        useInput,
                        None,
                        0,
                        repeatCounts,
                        simpleCore,
                        resNet,
                        preAct,
                        actUnit,
                        batchNorm
                    )
                )

                mainLayer.append(
                    GridUpCell(
                        ioShape[::-1],
                        noInput,
                        outShape,
                        repeatCounts,
                        simpleCore,
                        resNet,
                        preAct,
                        actUnit,
                        batchNorm
                    )
                )

                break

            elif altCounts == 1:
                mainLayer.append(
                    GridDownCell(
                        ioShape,
                        useInput,
                        outShape,
                        outCounts,
                        repeatCounts,
                        simpleCore,
                        resNet,
                        preAct,
                        actUnit,
                        batchNorm
                    )
                )

                break

            else:
                mainLayer.append(
                    GridDownCell(
                        ioShape,
                        useInput,
                        None,
                        0,
                        repeatCounts,
                        simpleCore,
                        resNet,
                        preAct,
                        actUnit,
                        batchNorm
                    )
                )

                mainLayer.append(
                    GridUpCell(
                        ioShape[::-1],
                        noInput,
                        None,
                        repeatCounts,
                        simpleCore,
                        resNet,
                        preAct,
                        actUnit,
                        batchNorm
                    )
                )

                altCounts -= 2

            useInput = False

        self.mainLayer = nn.ModuleList(mainLayer)

    def forward(self, x):
        for mainLayer in self.mainLayer:
            x = mainLayer(x)

        return x



class DUGridNet(nn.Module):

    def __init__(self):
        super().__init__()

        ioShape = [8, 16, 32, 64, 128]
        ioInit = (1, 8, "same")
        outShape = (1, 512)

        repeatCounts = 1
        altCounts = 4

        preAct = True
        actUnit = "ReLU"
        batchNorm = False
        simpleCore = True
        resNet = False

        # Keep the original attribute name and module hierarchy so that
        # the original checkpoint state_dict can be loaded directly.
        mainLayer = [
            InitialCNN(
                ioInit,
                preAct,
                actUnit,
                batchNorm
            ),
            GridCell(
                ioShape,
                True,
                outShape,
                0,
                repeatCounts,
                altCounts,
                simpleCore,
                resNet,
                preAct,
                actUnit,
                batchNorm
            )
        ]

        self.mainLayer = nn.ModuleList(mainLayer)

    def forward(self, x):
        for mainLayer in self.mainLayer:
            x = mainLayer(x)

        return x


def load_du_gridnet(checkpoint_path, device="cpu"):
    """
    Load a DU-GridNet checkpoint saved in the original format:

        torch.save({
            'modelState': model.state_dict(),
            ...
        }, path)

    Returns
    -------
    model : DUGridNet
        Model in evaluation mode.
    """
    device = torch.device(device)

    model = DUGridNet().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if "modelState" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'modelState'."
        )

    model.load_state_dict(
        checkpoint["modelState"],
        strict=True
    )

    model.eval()

    return model
