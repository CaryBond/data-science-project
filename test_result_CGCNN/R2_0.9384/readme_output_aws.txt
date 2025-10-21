command on AWS EC2 SSH:
cgcnn-tr -as cif/ --train-ratio 0.8 --valid-ratio 0.10 --test-ratio 0.10 --epoch
 300 --batch-size 32 --workers 10 --learning-rate 0.001 --task regression

training output or process:

2025-07-07 13:50:09.939 INFO: cgcnn2 version: 0.4.7
2025-07-07 13:50:09.939 INFO: cuda version: 12.6
2025-07-07 13:50:09.939 INFO: torch version: 2.7.1+cu126
2025-07-07 13:50:09.941 INFO: Parsed arguments:
{'atom_fea_len': 64,
 'axis_limits': None,
 'batch_size': 32,
 'bias_temperature': -1.0,
 'cache_size': None,
 'device': device(type='cpu'),
 'disable_cuda': False,
 'epoch': 300.0,
 'full_set': 'cif/',
 'h_fea_len': 128,
 'job_id': '147999',
 'learning_rate': 0.001,
 'lr_factor': 0.5,
 'lr_patience': 0,
 'n_conv': 3,
 'n_h': 1,
 'random_seed': 42,
 'stop_patience': None,
 'task': 'regression',
 'test_ratio': 0.1,
 'test_set': None,
 'train_force_set': None,
 'train_ratio': 0.8,
 'train_set': None,
 'valid_ratio': 0.1,
 'valid_set': None,
 'workers': 10,
 'xlabel': 'Actual',
 'ylabel': 'Predicted'}
2025-07-07 13:55:26.649 INFO: Epoch [001/300] - Train Loss: 145.13359, Valid Loss: 70.66399, LR: [0.001]
2025-07-07 13:55:26.653 INFO:   [SAVE] Best model at epoch 1.
2025-07-07 13:56:30.354 INFO: Epoch [002/300] - Train Loss: 71.42647, Valid Loss: 58.69089, LR: [0.001]
2025-07-07 13:56:30.357 INFO:   [SAVE] Best model at epoch 2.
2025-07-07 13:57:34.050 INFO: Epoch [003/300] - Train Loss: 62.00600, Valid Loss: 59.79501, LR: [0.001]
2025-07-07 13:58:37.899 INFO: Epoch [004/300] - Train Loss: 60.01820, Valid Loss: 125.75354, LR: [0.001]
2025-07-07 13:59:41.751 INFO: Epoch [005/300] - Train Loss: 52.35379, Valid Loss: 47.12104, LR: [0.001]
2025-07-07 13:59:41.754 INFO:   [SAVE] Best model at epoch 5.
2025-07-07 14:00:45.312 INFO: Epoch [006/300] - Train Loss: 53.96416, Valid Loss: 73.27421, LR: [0.001]
2025-07-07 14:01:49.133 INFO: Epoch [007/300] - Train Loss: 51.38559, Valid Loss: 85.76956, LR: [0.001]
2025-07-07 14:02:52.819 INFO: Epoch [008/300] - Train Loss: 48.92486, Valid Loss: 66.58531, LR: [0.001]
2025-07-07 14:03:56.507 INFO: Epoch [009/300] - Train Loss: 49.30426, Valid Loss: 59.09953, LR: [0.001]
2025-07-07 14:05:00.981 INFO: Epoch [010/300] - Train Loss: 45.36765, Valid Loss: 43.12831, LR: [0.001]
2025-07-07 14:05:00.984 INFO:   [SAVE] Best model at epoch 10.
2025-07-07 14:06:05.122 INFO: Epoch [011/300] - Train Loss: 47.13087, Valid Loss: 33.04504, LR: [0.001]
2025-07-07 14:06:05.125 INFO:   [SAVE] Best model at epoch 11.
2025-07-07 14:07:09.199 INFO: Epoch [012/300] - Train Loss: 44.91023, Valid Loss: 48.41712, LR: [0.001]
2025-07-07 14:08:12.840 INFO: Epoch [013/300] - Train Loss: 44.97479, Valid Loss: 70.31289, LR: [0.001]
2025-07-07 14:09:16.808 INFO: Epoch [014/300] - Train Loss: 42.39355, Valid Loss: 32.32551, LR: [0.001]
2025-07-07 14:09:16.811 INFO:   [SAVE] Best model at epoch 14.
2025-07-07 14:10:21.276 INFO: Epoch [015/300] - Train Loss: 43.08660, Valid Loss: 34.03443, LR: [0.001]
2025-07-07 14:11:25.244 INFO: Epoch [016/300] - Train Loss: 39.92392, Valid Loss: 60.44661, LR: [0.001]
2025-07-07 14:12:29.780 INFO: Epoch [017/300] - Train Loss: 41.85384, Valid Loss: 34.03939, LR: [0.001]
2025-07-07 14:13:34.114 INFO: Epoch [018/300] - Train Loss: 40.40917, Valid Loss: 39.61725, LR: [0.001]
2025-07-07 14:14:37.996 INFO: Epoch [019/300] - Train Loss: 38.88389, Valid Loss: 31.80690, LR: [0.001]
2025-07-07 14:14:37.999 INFO:   [SAVE] Best model at epoch 19.
2025-07-07 14:15:41.816 INFO: Epoch [020/300] - Train Loss: 38.35925, Valid Loss: 35.77852, LR: [0.001]
2025-07-07 14:16:45.697 INFO: Epoch [021/300] - Train Loss: 38.57522, Valid Loss: 41.49246, LR: [0.001]
2025-07-07 14:17:49.263 INFO: Epoch [022/300] - Train Loss: 36.85963, Valid Loss: 39.58520, LR: [0.001]
2025-07-07 14:18:53.221 INFO: Epoch [023/300] - Train Loss: 37.85691, Valid Loss: 33.11440, LR: [0.001]
2025-07-07 14:19:57.425 INFO: Epoch [024/300] - Train Loss: 35.85350, Valid Loss: 66.97440, LR: [0.001]
2025-07-07 14:21:00.907 INFO: Epoch [025/300] - Train Loss: 35.71207, Valid Loss: 37.44357, LR: [0.001]
2025-07-07 14:22:04.420 INFO: Epoch [026/300] - Train Loss: 38.71213, Valid Loss: 77.31801, LR: [0.001]
2025-07-07 14:23:07.955 INFO: Epoch [027/300] - Train Loss: 34.60798, Valid Loss: 27.84433, LR: [0.001]
2025-07-07 14:23:07.958 INFO:   [SAVE] Best model at epoch 27.
2025-07-07 14:24:11.327 INFO: Epoch [028/300] - Train Loss: 35.35787, Valid Loss: 44.23485, LR: [0.001]
2025-07-07 14:25:15.363 INFO: Epoch [029/300] - Train Loss: 35.42241, Valid Loss: 32.07328, LR: [0.001]
2025-07-07 14:26:18.458 INFO: Epoch [030/300] - Train Loss: 34.86999, Valid Loss: 27.99657, LR: [0.001]
2025-07-07 14:27:22.382 INFO: Epoch [031/300] - Train Loss: 31.97947, Valid Loss: 41.73594, LR: [0.001]
2025-07-07 14:28:26.523 INFO: Epoch [032/300] - Train Loss: 33.20963, Valid Loss: 27.64309, LR: [0.001]
2025-07-07 14:28:26.526 INFO:   [SAVE] Best model at epoch 32.
2025-07-07 14:29:30.626 INFO: Epoch [033/300] - Train Loss: 34.83153, Valid Loss: 31.28060, LR: [0.001]
2025-07-07 14:30:34.925 INFO: Epoch [034/300] - Train Loss: 33.21885, Valid Loss: 42.29312, LR: [0.001]
2025-07-07 14:31:38.732 INFO: Epoch [035/300] - Train Loss: 31.88239, Valid Loss: 46.70096, LR: [0.001]
2025-07-07 14:32:43.156 INFO: Epoch [036/300] - Train Loss: 32.59813, Valid Loss: 27.20475, LR: [0.001]
2025-07-07 14:32:43.160 INFO:   [SAVE] Best model at epoch 36.
2025-07-07 14:33:46.955 INFO: Epoch [037/300] - Train Loss: 30.85113, Valid Loss: 66.64733, LR: [0.001]
2025-07-07 14:34:51.142 INFO: Epoch [038/300] - Train Loss: 29.37677, Valid Loss: 25.12142, LR: [0.001]
2025-07-07 14:34:51.145 INFO:   [SAVE] Best model at epoch 38.
2025-07-07 14:35:55.459 INFO: Epoch [039/300] - Train Loss: 29.64249, Valid Loss: 37.31337, LR: [0.001]
2025-07-07 14:36:59.214 INFO: Epoch [040/300] - Train Loss: 29.69852, Valid Loss: 68.96697, LR: [0.001]
2025-07-07 14:38:03.210 INFO: Epoch [041/300] - Train Loss: 30.24724, Valid Loss: 24.92241, LR: [0.001]
2025-07-07 14:38:03.213 INFO:   [SAVE] Best model at epoch 41.
2025-07-07 14:39:07.396 INFO: Epoch [042/300] - Train Loss: 29.72729, Valid Loss: 38.04756, LR: [0.001]
2025-07-07 14:40:11.624 INFO: Epoch [043/300] - Train Loss: 28.41298, Valid Loss: 23.94783, LR: [0.001]
2025-07-07 14:40:11.627 INFO:   [SAVE] Best model at epoch 43.
2025-07-07 14:41:15.592 INFO: Epoch [044/300] - Train Loss: 28.05005, Valid Loss: 35.69772, LR: [0.001]
2025-07-07 14:42:19.527 INFO: Epoch [045/300] - Train Loss: 29.44916, Valid Loss: 26.35992, LR: [0.001]
2025-07-07 14:43:23.364 INFO: Epoch [046/300] - Train Loss: 27.98627, Valid Loss: 31.14407, LR: [0.001]
2025-07-07 14:44:27.081 INFO: Epoch [047/300] - Train Loss: 26.24479, Valid Loss: 21.75458, LR: [0.001]
2025-07-07 14:44:27.084 INFO:   [SAVE] Best model at epoch 47.
2025-07-07 14:45:31.006 INFO: Epoch [048/300] - Train Loss: 25.70601, Valid Loss: 38.32755, LR: [0.001]
2025-07-07 14:46:35.185 INFO: Epoch [049/300] - Train Loss: 26.62980, Valid Loss: 24.83777, LR: [0.001]
2025-07-07 14:47:39.232 INFO: Epoch [050/300] - Train Loss: 24.68568, Valid Loss: 24.68784, LR: [0.001]
2025-07-07 14:48:42.882 INFO: Epoch [051/300] - Train Loss: 25.49215, Valid Loss: 21.97870, LR: [0.001]
2025-07-07 14:49:46.897 INFO: Epoch [052/300] - Train Loss: 23.82297, Valid Loss: 23.99361, LR: [0.001]
2025-07-07 14:50:50.560 INFO: Epoch [053/300] - Train Loss: 23.30123, Valid Loss: 27.52566, LR: [0.001]
2025-07-07 14:51:54.306 INFO: Epoch [054/300] - Train Loss: 22.18455, Valid Loss: 27.22631, LR: [0.001]
2025-07-07 14:52:58.957 INFO: Epoch [055/300] - Train Loss: 23.57537, Valid Loss: 41.18471, LR: [0.001]
2025-07-07 14:54:02.827 INFO: Epoch [056/300] - Train Loss: 23.08720, Valid Loss: 62.34313, LR: [0.001]
2025-07-07 14:55:06.874 INFO: Epoch [057/300] - Train Loss: 26.24030, Valid Loss: 27.40375, LR: [0.001]
2025-07-07 14:56:11.205 INFO: Epoch [058/300] - Train Loss: 22.45360, Valid Loss: 51.73591, LR: [0.001]
2025-07-07 14:57:15.226 INFO: Epoch [059/300] - Train Loss: 22.61561, Valid Loss: 21.96210, LR: [0.001]
2025-07-07 14:58:19.263 INFO: Epoch [060/300] - Train Loss: 22.05188, Valid Loss: 43.26816, LR: [0.001]
2025-07-07 14:59:23.461 INFO: Epoch [061/300] - Train Loss: 22.11489, Valid Loss: 24.10661, LR: [0.001]
2025-07-07 15:00:27.840 INFO: Epoch [062/300] - Train Loss: 20.76165, Valid Loss: 24.50577, LR: [0.001]
2025-07-07 15:01:32.080 INFO: Epoch [063/300] - Train Loss: 20.21405, Valid Loss: 22.32660, LR: [0.001]
2025-07-07 15:02:36.488 INFO: Epoch [064/300] - Train Loss: 19.51825, Valid Loss: 19.23875, LR: [0.001]
2025-07-07 15:02:36.491 INFO:   [SAVE] Best model at epoch 64.
2025-07-07 15:03:40.374 INFO: Epoch [065/300] - Train Loss: 19.05706, Valid Loss: 33.80222, LR: [0.001]
2025-07-07 15:04:44.694 INFO: Epoch [066/300] - Train Loss: 20.86982, Valid Loss: 22.46579, LR: [0.001]
2025-07-07 15:05:48.276 INFO: Epoch [067/300] - Train Loss: 19.31237, Valid Loss: 23.12674, LR: [0.001]
2025-07-07 15:06:52.509 INFO: Epoch [068/300] - Train Loss: 19.89656, Valid Loss: 22.24508, LR: [0.001]
2025-07-07 15:07:56.891 INFO: Epoch [069/300] - Train Loss: 18.72458, Valid Loss: 39.37577, LR: [0.001]
2025-07-07 15:09:00.740 INFO: Epoch [070/300] - Train Loss: 18.97806, Valid Loss: 35.11708, LR: [0.001]
2025-07-07 15:10:04.710 INFO: Epoch [071/300] - Train Loss: 19.04204, Valid Loss: 34.26223, LR: [0.001]
2025-07-07 15:11:08.828 INFO: Epoch [072/300] - Train Loss: 17.79914, Valid Loss: 22.03926, LR: [0.001]
2025-07-07 15:12:12.803 INFO: Epoch [073/300] - Train Loss: 18.87640, Valid Loss: 19.28532, LR: [0.001]
2025-07-07 15:13:16.827 INFO: Epoch [074/300] - Train Loss: 17.67015, Valid Loss: 22.46142, LR: [0.001]
2025-07-07 15:14:20.545 INFO: Epoch [075/300] - Train Loss: 18.58242, Valid Loss: 20.03437, LR: [0.001]
2025-07-07 15:15:24.121 INFO: Epoch [076/300] - Train Loss: 17.74272, Valid Loss: 18.52139, LR: [0.001]
2025-07-07 15:15:24.124 INFO:   [SAVE] Best model at epoch 76.
2025-07-07 15:16:28.195 INFO: Epoch [077/300] - Train Loss: 16.09999, Valid Loss: 17.34038, LR: [0.001]
2025-07-07 15:16:28.198 INFO:   [SAVE] Best model at epoch 77.
2025-07-07 15:17:32.387 INFO: Epoch [078/300] - Train Loss: 17.50173, Valid Loss: 18.00840, LR: [0.001]
2025-07-07 15:18:36.523 INFO: Epoch [079/300] - Train Loss: 17.46456, Valid Loss: 37.95350, LR: [0.001]
2025-07-07 15:19:40.142 INFO: Epoch [080/300] - Train Loss: 16.56883, Valid Loss: 22.69624, LR: [0.001]
2025-07-07 15:20:44.148 INFO: Epoch [081/300] - Train Loss: 15.09897, Valid Loss: 16.04247, LR: [0.001]
2025-07-07 15:20:44.151 INFO:   [SAVE] Best model at epoch 81.
2025-07-07 15:21:48.082 INFO: Epoch [082/300] - Train Loss: 16.65228, Valid Loss: 24.43983, LR: [0.001]
2025-07-07 15:22:52.040 INFO: Epoch [083/300] - Train Loss: 14.86750, Valid Loss: 19.08828, LR: [0.001]
2025-07-07 15:23:56.211 INFO: Epoch [084/300] - Train Loss: 15.25154, Valid Loss: 20.56026, LR: [0.001]
2025-07-07 15:25:00.342 INFO: Epoch [085/300] - Train Loss: 16.98723, Valid Loss: 22.49440, LR: [0.001]
2025-07-07 15:26:04.297 INFO: Epoch [086/300] - Train Loss: 15.58703, Valid Loss: 24.16875, LR: [0.001]
2025-07-07 15:27:07.995 INFO: Epoch [087/300] - Train Loss: 15.54818, Valid Loss: 18.38323, LR: [0.001]
2025-07-07 15:28:12.025 INFO: Epoch [088/300] - Train Loss: 14.84541, Valid Loss: 29.64681, LR: [0.001]
2025-07-07 15:29:15.766 INFO: Epoch [089/300] - Train Loss: 14.85156, Valid Loss: 17.63624, LR: [0.001]
2025-07-07 15:30:19.360 INFO: Epoch [090/300] - Train Loss: 14.25665, Valid Loss: 18.96146, LR: [0.001]
2025-07-07 15:31:23.368 INFO: Epoch [091/300] - Train Loss: 13.63489, Valid Loss: 25.02355, LR: [0.001]
2025-07-07 15:32:27.268 INFO: Epoch [092/300] - Train Loss: 13.81440, Valid Loss: 20.31598, LR: [0.001]
2025-07-07 15:33:31.320 INFO: Epoch [093/300] - Train Loss: 14.51310, Valid Loss: 15.91245, LR: [0.001]
2025-07-07 15:33:31.322 INFO:   [SAVE] Best model at epoch 93.
2025-07-07 15:34:35.271 INFO: Epoch [094/300] - Train Loss: 14.85551, Valid Loss: 57.97665, LR: [0.001]
2025-07-07 15:35:39.033 INFO: Epoch [095/300] - Train Loss: 13.44530, Valid Loss: 20.55520, LR: [0.001]
2025-07-07 15:36:42.907 INFO: Epoch [096/300] - Train Loss: 14.40465, Valid Loss: 15.20565, LR: [0.001]
2025-07-07 15:36:42.910 INFO:   [SAVE] Best model at epoch 96.
2025-07-07 15:37:46.679 INFO: Epoch [097/300] - Train Loss: 13.77099, Valid Loss: 42.58617, LR: [0.001]
2025-07-07 15:38:50.482 INFO: Epoch [098/300] - Train Loss: 14.13465, Valid Loss: 44.41498, LR: [0.001]
2025-07-07 15:39:54.303 INFO: Epoch [099/300] - Train Loss: 13.34564, Valid Loss: 17.23938, LR: [0.001]
 2025-07-07 15:40:58.345 INFO: Epoch [100/300] - Train Loss: 14.56315, Valid Loss: 20.57485, LR: [0.001]
2025-07-07 15:42:02.406 INFO: Epoch [101/300] - Train Loss: 12.42158, Valid Loss: 20.28801, LR: [0.001]
2025-07-07 15:43:06.315 INFO: Epoch [102/300] - Train Loss: 12.65600, Valid Loss: 19.68588, LR: [0.001]
2025-07-07 15:44:10.266 INFO: Epoch [103/300] - Train Loss: 12.96900, Valid Loss: 22.77333, LR: [0.001]
2025-07-07 15:45:14.004 INFO: Epoch [104/300] - Train Loss: 12.93349, Valid Loss: 16.20165, LR: [0.001]
2025-07-07 15:46:18.026 INFO: Epoch [105/300] - Train Loss: 12.78356, Valid Loss: 19.89660, LR: [0.001]
2025-07-07 15:47:22.036 INFO: Epoch [106/300] - Train Loss: 12.83382, Valid Loss: 32.07401, LR: [0.001]
2025-07-07 15:48:25.918 INFO: Epoch [107/300] - Train Loss: 12.12463, Valid Loss: 47.26635, LR: [0.001]
2025-07-07 15:49:29.274 INFO: Epoch [108/300] - Train Loss: 12.46328, Valid Loss: 15.00204, LR: [0.001]
2025-07-07 15:49:29.277 INFO:   [SAVE] Best model at epoch 108.
2025-07-07 15:50:33.379 INFO: Epoch [109/300] - Train Loss: 11.68519, Valid Loss: 22.43864, LR: [0.001]
2025-07-07 15:51:37.218 INFO: Epoch [110/300] - Train Loss: 11.98040, Valid Loss: 15.84613, LR: [0.001]
2025-07-07 15:52:41.076 INFO: Epoch [111/300] - Train Loss: 12.52524, Valid Loss: 23.84729, LR: [0.001]
2025-07-07 15:53:45.287 INFO: Epoch [112/300] - Train Loss: 11.63777, Valid Loss: 17.89224, LR: [0.001]
2025-07-07 15:54:49.581 INFO: Epoch [113/300] - Train Loss: 12.21160, Valid Loss: 35.01603, LR: [0.001]
2025-07-07 15:55:53.720 INFO: Epoch [114/300] - Train Loss: 12.62778, Valid Loss: 13.81950, LR: [0.001]
2025-07-07 15:55:53.723 INFO:   [SAVE] Best model at epoch 114.
2025-07-07 15:56:57.870 INFO: Epoch [115/300] - Train Loss: 11.79528, Valid Loss: 20.65165, LR: [0.001]
2025-07-07 15:58:01.642 INFO: Epoch [116/300] - Train Loss: 11.28951, Valid Loss: 18.79479, LR: [0.001]
2025-07-07 15:59:06.003 INFO: Epoch [117/300] - Train Loss: 10.79233, Valid Loss: 16.29777, LR: [0.001]
2025-07-07 16:00:09.746 INFO: Epoch [118/300] - Train Loss: 11.51554, Valid Loss: 29.66043, LR: [0.001]
2025-07-07 16:01:13.924 INFO: Epoch [119/300] - Train Loss: 10.30252, Valid Loss: 22.81846, LR: [0.001]
2025-07-07 16:02:18.194 INFO: Epoch [120/300] - Train Loss: 11.19092, Valid Loss: 15.31869, LR: [0.001]
2025-07-07 16:03:22.316 INFO: Epoch [121/300] - Train Loss: 10.24921, Valid Loss: 14.88897, LR: [0.001]
2025-07-07 16:04:26.415 INFO: Epoch [122/300] - Train Loss: 10.19361, Valid Loss: 25.58213, LR: [0.001]
2025-07-07 16:05:30.388 INFO: Epoch [123/300] - Train Loss: 10.06862, Valid Loss: 15.21222, LR: [0.001]
2025-07-07 16:06:34.670 INFO: Epoch [124/300] - Train Loss: 10.85521, Valid Loss: 16.10813, LR: [0.001]
2025-07-07 16:07:38.668 INFO: Epoch [125/300] - Train Loss: 10.45074, Valid Loss: 16.70865, LR: [0.001]
2025-07-07 16:08:42.871 INFO: Epoch [126/300] - Train Loss: 9.99686, Valid Loss: 13.69753, LR: [0.001]
2025-07-07 16:08:42.874 INFO:   [SAVE] Best model at epoch 126.
2025-07-07 16:09:47.032 INFO: Epoch [127/300] - Train Loss: 10.29829, Valid Loss: 24.85502, LR: [0.001]
2025-07-07 16:10:51.139 INFO: Epoch [128/300] - Train Loss: 9.14016, Valid Loss: 15.89010, LR: [0.001]
2025-07-07 16:11:55.276 INFO: Epoch [129/300] - Train Loss: 10.03736, Valid Loss: 31.28208, LR: [0.001]
2025-07-07 16:12:59.300 INFO: Epoch [130/300] - Train Loss: 9.99548, Valid Loss: 19.18852, LR: [0.001]
2025-07-07 16:14:03.565 INFO: Epoch [131/300] - Train Loss: 10.10696, Valid Loss: 30.38542, LR: [0.001]
2025-07-07 16:15:07.614 INFO: Epoch [132/300] - Train Loss: 9.49157, Valid Loss: 17.54192, LR: [0.001]
2025-07-07 16:16:11.894 INFO: Epoch [133/300] - Train Loss: 9.74387, Valid Loss: 15.09136, LR: [0.001]
2025-07-07 16:17:15.721 INFO: Epoch [134/300] - Train Loss: 9.65063, Valid Loss: 22.17552, LR: [0.001]
2025-07-07 16:18:19.346 INFO: Epoch [135/300] - Train Loss: 9.19749, Valid Loss: 19.30409, LR: [0.001]
2025-07-07 16:19:23.376 INFO: Epoch [136/300] - Train Loss: 9.46220, Valid Loss: 18.62487, LR: [0.001]
2025-07-07 16:20:27.435 INFO: Epoch [137/300] - Train Loss: 9.79788, Valid Loss: 16.60266, LR: [0.001]
2025-07-07 16:21:31.547 INFO: Epoch [138/300] - Train Loss: 9.75474, Valid Loss: 16.39369, LR: [0.001]
2025-07-07 16:22:35.883 INFO: Epoch [139/300] - Train Loss: 8.83508, Valid Loss: 16.33928, LR: [0.001]
2025-07-07 16:23:40.395 INFO: Epoch [140/300] - Train Loss: 9.37074, Valid Loss: 15.71771, LR: [0.001]
2025-07-07 16:24:44.771 INFO: Epoch [141/300] - Train Loss: 8.54173, Valid Loss: 18.60880, LR: [0.001]
2025-07-07 16:25:49.023 INFO: Epoch [142/300] - Train Loss: 9.17528, Valid Loss: 14.04980, LR: [0.001]
2025-07-07 16:26:53.069 INFO: Epoch [143/300] - Train Loss: 8.08914, Valid Loss: 14.73563, LR: [0.001]
2025-07-07 16:27:57.300 INFO: Epoch [144/300] - Train Loss: 9.12986, Valid Loss: 17.72566, LR: [0.001]
2025-07-07 16:29:01.780 INFO: Epoch [145/300] - Train Loss: 8.51586, Valid Loss: 26.91421, LR: [0.001]
2025-07-07 16:30:06.080 INFO: Epoch [146/300] - Train Loss: 8.66853, Valid Loss: 13.91296, LR: [0.001]
2025-07-07 16:31:10.373 INFO: Epoch [147/300] - Train Loss: 8.73950, Valid Loss: 14.77414, LR: [0.001]
2025-07-07 16:32:14.734 INFO: Epoch [148/300] - Train Loss: 7.99433, Valid Loss: 19.19070, LR: [0.001]
2025-07-07 16:33:18.864 INFO: Epoch [149/300] - Train Loss: 7.64227, Valid Loss: 21.85151, LR: [0.001]
2025-07-07 16:34:22.917 INFO: Epoch [150/300] - Train Loss: 8.88189, Valid Loss: 14.10645, LR: [0.001]
2025-07-07 16:35:26.785 INFO: Epoch [151/300] - Train Loss: 8.98678, Valid Loss: 15.64518, LR: [0.001]
2025-07-07 16:36:31.075 INFO: Epoch [152/300] - Train Loss: 7.83737, Valid Loss: 40.59151, LR: [0.001]
2025-07-07 16:37:35.444 INFO: Epoch [153/300] - Train Loss: 8.83564, Valid Loss: 14.08313, LR: [0.001]
2025-07-07 16:38:39.661 INFO: Epoch [154/300] - Train Loss: 8.33030, Valid Loss: 16.16246, LR: [0.001]
2025-07-07 16:39:43.862 INFO: Epoch [155/300] - Train Loss: 8.35446, Valid Loss: 13.35185, LR: [0.001]
2025-07-07 16:39:43.865 INFO:   [SAVE] Best model at epoch 155.
2025-07-07 16:40:47.847 INFO: Epoch [156/300] - Train Loss: 7.52611, Valid Loss: 13.99328, LR: [0.001]
2025-07-07 16:41:51.828 INFO: Epoch [157/300] - Train Loss: 8.08703, Valid Loss: 15.78588, LR: [0.001]
2025-07-07 16:42:56.056 INFO: Epoch [158/300] - Train Loss: 7.34767, Valid Loss: 16.20543, LR: [0.001]
2025-07-07 16:44:00.085 INFO: Epoch [159/300] - Train Loss: 7.67077, Valid Loss: 13.92058, LR: [0.001]
2025-07-07 16:45:04.701 INFO: Epoch [160/300] - Train Loss: 7.51525, Valid Loss: 18.41607, LR: [0.001]
2025-07-07 16:46:08.679 INFO: Epoch [161/300] - Train Loss: 7.38630, Valid Loss: 15.62502, LR: [0.001]
2025-07-07 16:47:12.614 INFO: Epoch [162/300] - Train Loss: 7.29546, Valid Loss: 18.30364, LR: [0.001]
2025-07-07 16:48:16.833 INFO: Epoch [163/300] - Train Loss: 7.16233, Valid Loss: 13.48271, LR: [0.001]
2025-07-07 16:49:19.866 INFO: Epoch [164/300] - Train Loss: 7.59656, Valid Loss: 16.81465, LR: [0.001]
2025-07-07 16:50:23.904 INFO: Epoch [165/300] - Train Loss: 9.23774, Valid Loss: 21.96643, LR: [0.001]
2025-07-07 16:51:27.558 INFO: Epoch [166/300] - Train Loss: 6.65297, Valid Loss: 13.20084, LR: [0.001]
2025-07-07 16:51:27.561 INFO:   [SAVE] Best model at epoch 166.
2025-07-07 16:52:31.772 INFO: Epoch [167/300] - Train Loss: 7.43347, Valid Loss: 19.84570, LR: [0.001]
2025-07-07 16:53:35.710 INFO: Epoch [168/300] - Train Loss: 7.44795, Valid Loss: 17.30317, LR: [0.001]
2025-07-07 16:54:39.798 INFO: Epoch [169/300] - Train Loss: 8.37772, Valid Loss: 19.33410, LR: [0.001]
2025-07-07 16:55:43.930 INFO: Epoch [170/300] - Train Loss: 6.68062, Valid Loss: 12.80644, LR: [0.001]
2025-07-07 16:55:43.933 INFO:   [SAVE] Best model at epoch 170.
2025-07-07 16:56:48.012 INFO: Epoch [171/300] - Train Loss: 6.73776, Valid Loss: 13.31222, LR: [0.001]
2025-07-07 16:57:52.334 INFO: Epoch [172/300] - Train Loss: 6.59816, Valid Loss: 26.39149, LR: [0.001]
2025-07-07 16:58:55.962 INFO: Epoch [173/300] - Train Loss: 6.76874, Valid Loss: 19.34270, LR: [0.001]
2025-07-07 16:59:59.781 INFO: Epoch [174/300] - Train Loss: 7.15025, Valid Loss: 20.93894, LR: [0.001]
2025-07-07 17:01:03.278 INFO: Epoch [175/300] - Train Loss: 6.68152, Valid Loss: 17.87884, LR: [0.001]
2025-07-07 17:02:06.981 INFO: Epoch [176/300] - Train Loss: 6.83881, Valid Loss: 14.01448, LR: [0.001]
2025-07-07 17:03:10.875 INFO: Epoch [177/300] - Train Loss: 6.71780, Valid Loss: 13.11197, LR: [0.001]
2025-07-07 17:04:14.422 INFO: Epoch [178/300] - Train Loss: 6.73223, Valid Loss: 13.85457, LR: [0.001]
2025-07-07 17:05:17.838 INFO: Epoch [179/300] - Train Loss: 6.18530, Valid Loss: 12.65843, LR: [0.001]
2025-07-07 17:05:17.841 INFO:   [SAVE] Best model at epoch 179.
2025-07-07 17:06:21.341 INFO: Epoch [180/300] - Train Loss: 6.72921, Valid Loss: 19.17795, LR: [0.001]
2025-07-07 17:07:25.508 INFO: Epoch [181/300] - Train Loss: 7.38919, Valid Loss: 30.68528, LR: [0.001]
2025-07-07 17:08:29.745 INFO: Epoch [182/300] - Train Loss: 6.66051, Valid Loss: 16.66147, LR: [0.001]
2025-07-07 17:09:33.852 INFO: Epoch [183/300] - Train Loss: 6.80679, Valid Loss: 13.09959, LR: [0.001]
2025-07-07 17:10:38.063 INFO: Epoch [184/300] - Train Loss: 6.37779, Valid Loss: 20.99502, LR: [0.001]
2025-07-07 17:11:42.583 INFO: Epoch [185/300] - Train Loss: 6.24108, Valid Loss: 15.92600, LR: [0.001]
2025-07-07 17:12:46.776 INFO: Epoch [186/300] - Train Loss: 5.99092, Valid Loss: 16.81323, LR: [0.001]
2025-07-07 17:13:50.770 INFO: Epoch [187/300] - Train Loss: 5.79142, Valid Loss: 15.57282, LR: [0.001]
2025-07-07 17:14:54.745 INFO: Epoch [188/300] - Train Loss: 5.76424, Valid Loss: 19.31706, LR: [0.001]
2025-07-07 17:15:59.086 INFO: Epoch [189/300] - Train Loss: 6.54085, Valid Loss: 15.45024, LR: [0.001]
2025-07-07 17:17:03.166 INFO: Epoch [190/300] - Train Loss: 6.41153, Valid Loss: 16.82153, LR: [0.001]
2025-07-07 17:18:07.163 INFO: Epoch [191/300] - Train Loss: 6.56826, Valid Loss: 14.29584, LR: [0.001]
2025-07-07 17:19:11.188 INFO: Epoch [192/300] - Train Loss: 5.71915, Valid Loss: 13.90211, LR: [0.001]
2025-07-07 17:20:15.308 INFO: Epoch [193/300] - Train Loss: 6.23322, Valid Loss: 13.71290, LR: [0.001]
2025-07-07 17:21:19.723 INFO: Epoch [194/300] - Train Loss: 6.02822, Valid Loss: 21.74313, LR: [0.001]
2025-07-07 17:22:24.071 INFO: Epoch [195/300] - Train Loss: 6.18595, Valid Loss: 14.11680, LR: [0.001]
2025-07-07 17:23:28.600 INFO: Epoch [196/300] - Train Loss: 6.19879, Valid Loss: 17.13207, LR: [0.001]
2025-07-07 17:24:32.839 INFO: Epoch [197/300] - Train Loss: 5.74060, Valid Loss: 12.63069, LR: [0.001]
2025-07-07 17:24:32.842 INFO:   [SAVE] Best model at epoch 197.
2025-07-07 17:25:36.815 INFO: Epoch [198/300] - Train Loss: 6.51923, Valid Loss: 13.06056, LR: [0.001]
2025-07-07 17:26:40.973 INFO: Epoch [199/300] - Train Loss: 5.72709, Valid Loss: 13.58706, LR: [0.001]
2025-07-07 17:27:45.182 INFO: Epoch [200/300] - Train Loss: 6.20215, Valid Loss: 14.82695, LR: [0.001]
2025-07-07 17:28:49.559 INFO: Epoch [201/300] - Train Loss: 5.93212, Valid Loss: 13.98330, LR: [0.001]
2025-07-07 17:29:53.671 INFO: Epoch [202/300] - Train Loss: 5.73831, Valid Loss: 13.86755, LR: [0.001]
2025-07-07 17:30:57.847 INFO: Epoch [203/300] - Train Loss: 5.84237, Valid Loss: 13.31787, LR: [0.001]
2025-07-07 17:32:02.063 INFO: Epoch [204/300] - Train Loss: 5.48852, Valid Loss: 13.74410, LR: [0.001]
2025-07-07 17:33:06.448 INFO: Epoch [205/300] - Train Loss: 5.32230, Valid Loss: 14.55228, LR: [0.001]
2025-07-07 17:34:11.117 INFO: Epoch [206/300] - Train Loss: 5.84360, Valid Loss: 12.85112, LR: [0.001]
2025-07-07 17:35:15.156 INFO: Epoch [207/300] - Train Loss: 6.27017, Valid Loss: 19.59583, LR: [0.001]
2025-07-07 17:36:19.494 INFO: Epoch [208/300] - Train Loss: 5.29172, Valid Loss: 12.49125, LR: [0.001]
2025-07-07 17:36:19.497 INFO:   [SAVE] Best model at epoch 208.
2025-07-07 17:37:23.377 INFO: Epoch [209/300] - Train Loss: 5.27286, Valid Loss: 14.77316, LR: [0.001]
2025-07-07 17:38:28.183 INFO: Epoch [210/300] - Train Loss: 4.94021, Valid Loss: 14.14402, LR: [0.001]
2025-07-07 17:39:32.666 INFO: Epoch [211/300] - Train Loss: 6.08865, Valid Loss: 24.15826, LR: [0.001]
2025-07-07 17:40:36.923 INFO: Epoch [212/300] - Train Loss: 5.44277, Valid Loss: 15.01521, LR: [0.001]
2025-07-07 17:41:40.955 INFO: Epoch [213/300] - Train Loss: 5.38280, Valid Loss: 16.14956, LR: [0.001]
2025-07-07 17:42:45.606 INFO: Epoch [214/300] - Train Loss: 5.17951, Valid Loss: 13.49825, LR: [0.001]
2025-07-07 17:43:49.712 INFO: Epoch [215/300] - Train Loss: 4.83089, Valid Loss: 13.63514, LR: [0.001]
2025-07-07 17:44:53.995 INFO: Epoch [216/300] - Train Loss: 5.69792, Valid Loss: 37.77975, LR: [0.001]
2025-07-07 17:45:58.484 INFO: Epoch [217/300] - Train Loss: 5.55910, Valid Loss: 14.46975, LR: [0.001]
2025-07-07 17:47:02.847 INFO: Epoch [218/300] - Train Loss: 5.02116, Valid Loss: 12.83448, LR: [0.001]
2025-07-07 17:48:07.156 INFO: Epoch [219/300] - Train Loss: 5.08118, Valid Loss: 14.80661, LR: [0.001]
2025-07-07 17:49:11.341 INFO: Epoch [220/300] - Train Loss: 6.35564, Valid Loss: 32.56917, LR: [0.001]
2025-07-07 17:50:15.527 INFO: Epoch [221/300] - Train Loss: 5.39540, Valid Loss: 12.13902, LR: [0.001]
2025-07-07 17:50:15.530 INFO:   [SAVE] Best model at epoch 221.
2025-07-07 17:51:19.822 INFO: Epoch [222/300] - Train Loss: 4.89913, Valid Loss: 14.46564, LR: [0.001]
2025-07-07 17:52:22.929 INFO: Epoch [223/300] - Train Loss: 5.34298, Valid Loss: 14.44591, LR: [0.001]
2025-07-07 17:53:27.020 INFO: Epoch [224/300] - Train Loss: 5.34699, Valid Loss: 14.67329, LR: [0.001]
2025-07-07 17:54:31.088 INFO: Epoch [225/300] - Train Loss: 5.83521, Valid Loss: 18.27209, LR: [0.001]
2025-07-07 17:55:35.146 INFO: Epoch [226/300] - Train Loss: 4.96411, Valid Loss: 14.41296, LR: [0.001]
2025-07-07 17:56:39.395 INFO: Epoch [227/300] - Train Loss: 4.79381, Valid Loss: 13.19687, LR: [0.001]
2025-07-07 17:57:43.731 INFO: Epoch [228/300] - Train Loss: 4.84627, Valid Loss: 13.03589, LR: [0.001]
2025-07-07 17:58:47.897 INFO: Epoch [229/300] - Train Loss: 4.90746, Valid Loss: 16.05617, LR: [0.001]
2025-07-07 17:59:51.942 INFO: Epoch [230/300] - Train Loss: 4.89364, Valid Loss: 12.43668, LR: [0.001]
2025-07-07 18:00:56.063 INFO: Epoch [231/300] - Train Loss: 4.89755, Valid Loss: 14.52485, LR: [0.001]
2025-07-07 18:02:00.227 INFO: Epoch [232/300] - Train Loss: 5.80458, Valid Loss: 45.38590, LR: [0.001]
2025-07-07 18:03:04.190 INFO: Epoch [233/300] - Train Loss: 5.39409, Valid Loss: 15.07519, LR: [0.001]
2025-07-07 18:04:08.212 INFO: Epoch [234/300] - Train Loss: 4.91742, Valid Loss: 12.89182, LR: [0.001]
2025-07-07 18:05:12.764 INFO: Epoch [235/300] - Train Loss: 4.29583, Valid Loss: 13.31738, LR: [0.001]
2025-07-07 18:06:17.050 INFO: Epoch [236/300] - Train Loss: 4.87127, Valid Loss: 14.61679, LR: [0.001]
2025-07-07 18:07:21.258 INFO: Epoch [237/300] - Train Loss: 4.87534, Valid Loss: 18.75029, LR: [0.001]
2025-07-07 18:08:25.578 INFO: Epoch [238/300] - Train Loss: 4.70212, Valid Loss: 13.16985, LR: [0.001]
2025-07-07 18:09:29.876 INFO: Epoch [239/300] - Train Loss: 4.57058, Valid Loss: 12.90930, LR: [0.001]
2025-07-07 18:10:33.347 INFO: Epoch [240/300] - Train Loss: 5.42936, Valid Loss: 14.47580, LR: [0.001]
2025-07-07 18:11:37.783 INFO: Epoch [241/300] - Train Loss: 4.65829, Valid Loss: 14.02781, LR: [0.001]
2025-07-07 18:12:41.812 INFO: Epoch [242/300] - Train Loss: 4.59965, Valid Loss: 13.70459, LR: [0.001]
2025-07-07 18:13:46.123 INFO: Epoch [243/300] - Train Loss: 4.77983, Valid Loss: 12.96319, LR: [0.001]
2025-07-07 18:14:50.618 INFO: Epoch [244/300] - Train Loss: 4.90809, Valid Loss: 20.15697, LR: [0.001]
2025-07-07 18:15:55.056 INFO: Epoch [245/300] - Train Loss: 4.90109, Valid Loss: 15.57397, LR: [0.001]
2025-07-07 18:16:59.283 INFO: Epoch [246/300] - Train Loss: 4.84882, Valid Loss: 14.52288, LR: [0.001]
2025-07-07 18:18:03.489 INFO: Epoch [247/300] - Train Loss: 4.97066, Valid Loss: 13.11909, LR: [0.001]
2025-07-07 18:19:07.707 INFO: Epoch [248/300] - Train Loss: 4.25143, Valid Loss: 13.15417, LR: [0.001]
2025-07-07 18:20:11.756 INFO: Epoch [249/300] - Train Loss: 4.32299, Valid Loss: 11.86843, LR: [0.001]
2025-07-07 18:20:11.759 INFO:   [SAVE] Best model at epoch 249.
2025-07-07 18:21:16.176 INFO: Epoch [250/300] - Train Loss: 4.61765, Valid Loss: 13.01405, LR: [0.001]
2025-07-07 18:22:20.126 INFO: Epoch [251/300] - Train Loss: 4.97622, Valid Loss: 20.25168, LR: [0.001]
2025-07-07 18:23:24.058 INFO: Epoch [252/300] - Train Loss: 4.82882, Valid Loss: 15.58169, LR: [0.001]
2025-07-07 18:24:28.149 INFO: Epoch [253/300] - Train Loss: 3.96784, Valid Loss: 11.38385, LR: [0.001]
2025-07-07 18:24:28.151 INFO:   [SAVE] Best model at epoch 253.
2025-07-07 18:25:32.101 INFO: Epoch [254/300] - Train Loss: 4.10472, Valid Loss: 12.08409, LR: [0.001]
2025-07-07 18:26:36.432 INFO: Epoch [255/300] - Train Loss: 4.38072, Valid Loss: 12.20705, LR: [0.001]
2025-07-07 18:27:40.205 INFO: Epoch [256/300] - Train Loss: 4.50078, Valid Loss: 14.50773, LR: [0.001]
2025-07-07 18:28:43.832 INFO: Epoch [257/300] - Train Loss: 4.67113, Valid Loss: 13.14580, LR: [0.001]
2025-07-07 18:29:47.701 INFO: Epoch [258/300] - Train Loss: 3.88346, Valid Loss: 13.52790, LR: [0.001]
2025-07-07 18:30:51.809 INFO: Epoch [259/300] - Train Loss: 4.08810, Valid Loss: 13.73079, LR: [0.001]
2025-07-07 18:31:56.052 INFO: Epoch [260/300] - Train Loss: 4.33399, Valid Loss: 12.66288, LR: [0.001]
2025-07-07 18:32:59.829 INFO: Epoch [261/300] - Train Loss: 4.28890, Valid Loss: 13.37801, LR: [0.001]
2025-07-07 18:34:03.898 INFO: Epoch [262/300] - Train Loss: 4.09459, Valid Loss: 12.60432, LR: [0.001]
2025-07-07 18:35:08.310 INFO: Epoch [263/300] - Train Loss: 4.49480, Valid Loss: 14.04126, LR: [0.001]
2025-07-07 18:36:12.246 INFO: Epoch [264/300] - Train Loss: 3.95800, Valid Loss: 14.57832, LR: [0.001]
2025-07-07 18:37:16.441 INFO: Epoch [265/300] - Train Loss: 4.05880, Valid Loss: 10.98863, LR: [0.001]
2025-07-07 18:37:16.443 INFO:   [SAVE] Best model at epoch 265.
2025-07-07 18:38:20.587 INFO: Epoch [266/300] - Train Loss: 4.13730, Valid Loss: 13.66839, LR: [0.001]
2025-07-07 18:39:25.315 INFO: Epoch [267/300] - Train Loss: 3.94901, Valid Loss: 13.23089, LR: [0.001]
2025-07-07 18:40:29.598 INFO: Epoch [268/300] - Train Loss: 4.62291, Valid Loss: 11.51526, LR: [0.001]
2025-07-07 18:41:33.916 INFO: Epoch [269/300] - Train Loss: 4.15837, Valid Loss: 14.12917, LR: [0.001]
2025-07-07 18:42:37.662 INFO: Epoch [270/300] - Train Loss: 3.96509, Valid Loss: 14.64781, LR: [0.001]
2025-07-07 18:43:41.186 INFO: Epoch [271/300] - Train Loss: 4.01831, Valid Loss: 12.98025, LR: [0.001]
2025-07-07 18:44:45.232 INFO: Epoch [272/300] - Train Loss: 4.03801, Valid Loss: 11.71612, LR: [0.001]
2025-07-07 18:45:49.353 INFO: Epoch [273/300] - Train Loss: 4.07453, Valid Loss: 14.74103, LR: [0.001]
2025-07-07 18:46:53.445 INFO: Epoch [274/300] - Train Loss: 3.74431, Valid Loss: 15.23417, LR: [0.001]
2025-07-07 18:47:58.029 INFO: Epoch [275/300] - Train Loss: 3.89548, Valid Loss: 11.37821, LR: [0.001]
2025-07-07 18:49:02.108 INFO: Epoch [276/300] - Train Loss: 3.54446, Valid Loss: 12.96140, LR: [0.001]
2025-07-07 18:50:06.292 INFO: Epoch [277/300] - Train Loss: 3.90550, Valid Loss: 12.17476, LR: [0.001]
2025-07-07 18:51:10.492 INFO: Epoch [278/300] - Train Loss: 3.47533, Valid Loss: 11.52920, LR: [0.001]
2025-07-07 18:52:14.808 INFO: Epoch [279/300] - Train Loss: 3.63507, Valid Loss: 13.01607, LR: [0.001]
2025-07-07 18:53:19.006 INFO: Epoch [280/300] - Train Loss: 3.79603, Valid Loss: 16.69341, LR: [0.001]
2025-07-07 18:54:23.180 INFO: Epoch [281/300] - Train Loss: 4.53866, Valid Loss: 13.56251, LR: [0.001]
2025-07-07 18:55:26.934 INFO: Epoch [282/300] - Train Loss: 3.65585, Valid Loss: 14.37313, LR: [0.001]
2025-07-07 18:56:30.974 INFO: Epoch [283/300] - Train Loss: 3.80351, Valid Loss: 14.73777, LR: [0.001]
2025-07-07 18:57:35.056 INFO: Epoch [284/300] - Train Loss: 3.84066, Valid Loss: 13.70622, LR: [0.001]
2025-07-07 18:58:39.305 INFO: Epoch [285/300] - Train Loss: 3.97903, Valid Loss: 21.04951, LR: [0.001]
2025-07-07 18:59:43.269 INFO: Epoch [286/300] - Train Loss: 3.83179, Valid Loss: 14.26210, LR: [0.001]
2025-07-07 19:00:47.243 INFO: Epoch [287/300] - Train Loss: 3.65012, Valid Loss: 11.74408, LR: [0.001]
2025-07-07 19:01:51.368 INFO: Epoch [288/300] - Train Loss: 3.85269, Valid Loss: 20.15083, LR: [0.001]
2025-07-07 19:02:55.618 INFO: Epoch [289/300] - Train Loss: 4.47163, Valid Loss: 13.16619, LR: [0.001]
2025-07-07 19:03:59.355 INFO: Epoch [290/300] - Train Loss: 3.90716, Valid Loss: 17.18394, LR: [0.001]
2025-07-07 19:05:03.306 INFO: Epoch [291/300] - Train Loss: 4.46089, Valid Loss: 14.53138, LR: [0.001]
2025-07-07 19:06:07.582 INFO: Epoch [292/300] - Train Loss: 3.52413, Valid Loss: 13.08048, LR: [0.001]
2025-07-07 19:07:11.825 INFO: Epoch [293/300] - Train Loss: 3.55121, Valid Loss: 12.01153, LR: [0.001]
2025-07-07 19:08:15.824 INFO: Epoch [294/300] - Train Loss: 3.68457, Valid Loss: 12.06859, LR: [0.001]
2025-07-07 19:09:19.647 INFO: Epoch [295/300] - Train Loss: 3.45721, Valid Loss: 14.16796, LR: [0.001]
2025-07-07 19:10:23.792 INFO: Epoch [296/300] - Train Loss: 3.28994, Valid Loss: 16.36813, LR: [0.001]
2025-07-07 19:11:27.912 INFO: Epoch [297/300] - Train Loss: 3.67102, Valid Loss: 12.68982, LR: [0.001]
2025-07-07 19:12:31.858 INFO: Epoch [298/300] - Train Loss: 3.55786, Valid Loss: 14.84014, LR: [0.001]
2025-07-07 19:13:36.106 INFO: Epoch [299/300] - Train Loss: 3.43734, Valid Loss: 12.36274, LR: [0.001]
2025-07-07 19:14:40.172 INFO: Epoch [300/300] - Train Loss: 3.85249, Valid Loss: 21.12266, LR: [0.001]
2025-07-07 19:14:40.172 INFO: Training complete.
2025-07-07 19:14:42.354 INFO: MSE: 14.2267, R2 Score: 0.9384
2025-07-07 19:14:42.357 INFO: Prediction results have been saved to output_147999/test_results.csv

PyCharm: 
MSE: 14.2267
R²: 0.9384
MAE: 2.0564
RMSE: 3.7718
MedAE:  1.236959