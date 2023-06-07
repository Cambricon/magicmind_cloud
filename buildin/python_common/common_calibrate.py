import magicmind.python.runtime as mm

from logger import Logger

log = Logger()


def common_calibrate(args, network, config, calib_data):
    # network : mm.Network
    # config : mm.BuilderConfig
    # calib_data: instance of class CalibData
    # 创建量化工具并设置量化统计算法
    # calib_data = CalibData(shape = input_dims, max_samples = args.max_sample, img_dir = args.image_dir)
    calibrator = mm.Calibrator([calib_data])
    assert calibrator is not None
    # 设置量化统计算法，支持线性统计算法（LINEAR_ALGORITHM）及加强的最小化量化噪声算法（EQNM_ALGORITHM）。
    assert args.calibration_algo is not None, "calibration algorithm is not specified."
    PARAM_QUANT_ALGO_2_MM = {
        "linear": mm.QuantizationAlgorithm.LINEAR_ALGORITHM,
        "eqnm": mm.QuantizationAlgorithm.EQNM_ALGORITHM,
    }
    assert calibrator.set_quantization_algorithm(
        PARAM_QUANT_ALGO_2_MM[args.calibration_algo]
    ).ok()
    log.info("calibration: set quantization algotithm successfully.")
    # 打开设备
    with mm.System() as mm_sys:
        dev_count = mm_sys.device_count()
        log.info("Device count:{}".format(dev_count))
        if args.device_id >= dev_count:
            log.info("Invalid device set!")
            abort()
        # 打开MLU设备
        dev = mm.Device()
        dev.id = args.device_id
        assert dev.active().ok()
    # 进行量化
    assert calibrator.calibrate(network, config).ok()
