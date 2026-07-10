(.venv) kermit@kermit-PC-ART:~/Github/UAV-Search-Primitives_actions$ python scripts/train.py \
  --algo ddqn \
  --episodes 1000 \
  --device cuda \
  --train-every 4 \
  --learning-starts 1000 \
  --eval-every 50 \
  --eval-episodes 10 \
  --seed 42 \
  --run-dir runs/final/ddqn_seed42_1000
Using device: cuda
torch threads: 1
train_every: 4, learning_starts: 1000
DDQN primitive:   5%|▉                 | 49/1000 [00:10<03:40,  4.31it/s, reward=0.50, det=1.36, comp=0.78, best=-1000000000000000000.00, eps=0.64]
[Best] episode=50 score=-1.094 metrics={'eval_reward': -2.7944999999999984, 'eval_detected': 0.9, 'eval_completed': 0.4, 'eval_detected_value': 1.1, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.15099999941885472, 'stay_ratio': 0.6166666666666667, 'boundary_hit_ratio': 0.24133333333333334, 'revisit_ratio': 0.994}

[Eval 50] {'eval_reward': -2.7944999999999984, 'eval_detected': 0.9, 'eval_completed': 0.4, 'eval_detected_value': 1.1, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.15099999941885472, 'stay_ratio': 0.6166666666666667, 'boundary_hit_ratio': 0.24133333333333334, 'revisit_ratio': 0.994}
DDQN primitive:  10%|███▍                               | 99/1000 [00:26<07:32,  1.99it/s, reward=-0.98, det=1.34, comp=0.60, best=-1.09, eps=0.29]
[Eval 100] {'eval_reward': -4.076499999999999, 'eval_detected': 1.3, 'eval_completed': 0.5, 'eval_detected_value': 1.4, 'eval_completed_value': 0.5, 'eval_sensor_coverage': 0.24800000339746475, 'stay_ratio': 0.29533333333333334, 'boundary_hit_ratio': 0.448, 'revisit_ratio': 0.994}
DDQN primitive:  15%|█████                             | 149/1000 [01:01<10:15,  1.38it/s, reward=-2.09, det=1.12, comp=0.42, best=-1.09, eps=0.05]
[Best] episode=150 score=-0.544 metrics={'eval_reward': -2.2439999999999984, 'eval_detected': 1.1, 'eval_completed': 0.3, 'eval_detected_value': 1.3, 'eval_completed_value': 0.3, 'eval_sensor_coverage': 0.15149999894201754, 'stay_ratio': 0.688, 'boundary_hit_ratio': 0.14866666666666667, 'revisit_ratio': 0.9933333333333333}

[Eval 150] {'eval_reward': -2.2439999999999984, 'eval_detected': 1.1, 'eval_completed': 0.3, 'eval_detected_value': 1.3, 'eval_completed_value': 0.3, 'eval_sensor_coverage': 0.15149999894201754, 'stay_ratio': 0.688, 'boundary_hit_ratio': 0.14866666666666667, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  20%|██████▊                           | 199/1000 [01:37<08:34,  1.56it/s, reward=-2.31, det=1.14, comp=0.44, best=-0.54, eps=0.05]
[Best] episode=200 score=-0.533 metrics={'eval_reward': -2.5329999999999986, 'eval_detected': 1.2, 'eval_completed': 0.4, 'eval_detected_value': 1.2, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.2202499970793724, 'stay_ratio': 0.5906666666666667, 'boundary_hit_ratio': 0.202, 'revisit_ratio': 0.9933333333333333}

[Eval 200] {'eval_reward': -2.5329999999999986, 'eval_detected': 1.2, 'eval_completed': 0.4, 'eval_detected_value': 1.2, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.2202499970793724, 'stay_ratio': 0.5906666666666667, 'boundary_hit_ratio': 0.202, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  25%|████████▍                         | 249/1000 [02:12<08:09,  1.53it/s, reward=-2.59, det=0.86, comp=0.30, best=-0.53, eps=0.05]
[Best] episode=250 score=-0.424 metrics={'eval_reward': -3.1240000000000014, 'eval_detected': 1.7, 'eval_completed': 0.5, 'eval_detected_value': 1.9, 'eval_completed_value': 0.5, 'eval_sensor_coverage': 0.2764999970793724, 'stay_ratio': 0.21933333333333332, 'boundary_hit_ratio': 0.34933333333333333, 'revisit_ratio': 0.9933333333333333}

[Eval 250] {'eval_reward': -3.1240000000000014, 'eval_detected': 1.7, 'eval_completed': 0.5, 'eval_detected_value': 1.9, 'eval_completed_value': 0.5, 'eval_sensor_coverage': 0.2764999970793724, 'stay_ratio': 0.21933333333333332, 'boundary_hit_ratio': 0.34933333333333333, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  30%|██████████▏                       | 299/1000 [02:47<07:37,  1.53it/s, reward=-2.24, det=0.90, comp=0.38, best=-0.42, eps=0.05]
[Eval 300] {'eval_reward': -3.4529999999999994, 'eval_detected': 0.6, 'eval_completed': 0.1, 'eval_detected_value': 0.6, 'eval_completed_value': 0.1, 'eval_sensor_coverage': 0.17025000005960464, 'stay_ratio': 0.3, 'boundary_hit_ratio': 0.21466666666666667, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  35%|███████████▊                      | 349/1000 [03:23<07:03,  1.54it/s, reward=-2.31, det=0.98, comp=0.28, best=-0.42, eps=0.05]
[Eval 350] {'eval_reward': -2.697, 'eval_detected': 0.8, 'eval_completed': 0.2, 'eval_detected_value': 0.9, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.2545000024139881, 'stay_ratio': 0.32, 'boundary_hit_ratio': 0.16066666666666668, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  40%|█████████████▌                    | 399/1000 [03:58<06:33,  1.53it/s, reward=-2.69, det=1.00, comp=0.36, best=-0.42, eps=0.05]
[Eval 400] {'eval_reward': -2.9979999999999998, 'eval_detected': 0.7, 'eval_completed': 0.0, 'eval_detected_value': 0.8, 'eval_completed_value': 0.0, 'eval_sensor_coverage': 0.234749998152256, 'stay_ratio': 0.6126666666666667, 'boundary_hit_ratio': 0.13866666666666666, 'revisit_ratio': 0.996}
DDQN primitive:  45%|███████████████▎                  | 449/1000 [04:33<06:04,  1.51it/s, reward=-2.16, det=1.14, comp=0.34, best=-0.42, eps=0.05]
[Eval 450] {'eval_reward': -3.6879999999999953, 'eval_detected': 0.9, 'eval_completed': 0.2, 'eval_detected_value': 1.0, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.2647499985992908, 'stay_ratio': 0.36933333333333335, 'boundary_hit_ratio': 0.294, 'revisit_ratio': 0.996}
DDQN primitive:  50%|████████████████▉                 | 499/1000 [05:09<05:31,  1.51it/s, reward=-2.19, det=1.06, comp=0.44, best=-0.42, eps=0.05]
[Eval 500] {'eval_reward': -2.582999999999999, 'eval_detected': 1.2, 'eval_completed': 0.2, 'eval_detected_value': 1.5, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.2804999977350235, 'stay_ratio': 0.5153333333333333, 'boundary_hit_ratio': 0.17933333333333334, 'revisit_ratio': 0.9946666666666667}
DDQN primitive:  55%|██████████████████▋               | 549/1000 [05:44<04:58,  1.51it/s, reward=-1.92, det=1.10, comp=0.28, best=-0.42, eps=0.05]
[Eval 550] {'eval_reward': -3.1984999999999983, 'eval_detected': 1.4, 'eval_completed': 0.1, 'eval_detected_value': 1.8, 'eval_completed_value': 0.1, 'eval_sensor_coverage': 0.27574999928474425, 'stay_ratio': 0.346, 'boundary_hit_ratio': 0.25266666666666665, 'revisit_ratio': 0.994}
DDQN primitive:  60%|████████████████████▎             | 599/1000 [06:19<04:25,  1.51it/s, reward=-1.89, det=1.32, comp=0.38, best=-0.42, eps=0.05]
[Eval 600] {'eval_reward': -2.790000000000002, 'eval_detected': 1.2, 'eval_completed': 0.1, 'eval_detected_value': 1.6, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.2610000014305115, 'stay_ratio': 0.36666666666666664, 'boundary_hit_ratio': 0.274, 'revisit_ratio': 0.9946666666666667}
DDQN primitive:  65%|██████████████████████            | 649/1000 [06:55<03:55,  1.49it/s, reward=-1.65, det=1.16, comp=0.38, best=-0.42, eps=0.05]
[Eval 650] {'eval_reward': -4.545000000000008, 'eval_detected': 1.4, 'eval_completed': 0.3, 'eval_detected_value': 1.7, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.2885000005364418, 'stay_ratio': 0.15466666666666667, 'boundary_hit_ratio': 0.56, 'revisit_ratio': 0.9946666666666667}
DDQN primitive:  70%|███████████████████████▊          | 699/1000 [07:30<03:22,  1.49it/s, reward=-2.70, det=1.04, comp=0.30, best=-0.42, eps=0.05]
[Eval 700] {'eval_reward': -2.7730000000000006, 'eval_detected': 1.4, 'eval_completed': 0.3, 'eval_detected_value': 1.9, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.31950000375509263, 'stay_ratio': 0.2633333333333333, 'boundary_hit_ratio': 0.342, 'revisit_ratio': 0.9946666666666667}
DDQN primitive:  75%|█████████████████████████▍        | 749/1000 [08:06<02:47,  1.50it/s, reward=-0.80, det=1.36, comp=0.58, best=-0.42, eps=0.05]
[Best] episode=750 score=0.290 metrics={'eval_reward': -1.9100000000000026, 'eval_detected': 1.2, 'eval_completed': 0.5, 'eval_detected_value': 1.5, 'eval_completed_value': 0.6, 'eval_sensor_coverage': 0.22524999976158142, 'stay_ratio': 0.38666666666666666, 'boundary_hit_ratio': 0.256, 'revisit_ratio': 0.9946666666666667}

[Eval 750] {'eval_reward': -1.9100000000000026, 'eval_detected': 1.2, 'eval_completed': 0.5, 'eval_detected_value': 1.5, 'eval_completed_value': 0.6, 'eval_sensor_coverage': 0.22524999976158142, 'stay_ratio': 0.38666666666666666, 'boundary_hit_ratio': 0.256, 'revisit_ratio': 0.9946666666666667}
DDQN primitive:  80%|███████████████████████████▉       | 799/1000 [08:41<02:11,  1.53it/s, reward=-2.34, det=1.02, comp=0.30, best=0.29, eps=0.05]
[Eval 800] {'eval_reward': -3.0940000000000047, 'eval_detected': 1.5, 'eval_completed': 0.6, 'eval_detected_value': 2.0, 'eval_completed_value': 0.7, 'eval_sensor_coverage': 0.23775000125169754, 'stay_ratio': 0.17133333333333334, 'boundary_hit_ratio': 0.476, 'revisit_ratio': 0.9946666666666667}
DDQN primitive:  85%|█████████████████████████████▋     | 849/1000 [09:16<01:35,  1.59it/s, reward=-2.32, det=1.06, comp=0.38, best=0.29, eps=0.05]
[Eval 850] {'eval_reward': -3.4495000000000005, 'eval_detected': 1.4, 'eval_completed': 0.3, 'eval_detected_value': 1.6, 'eval_completed_value': 0.3, 'eval_sensor_coverage': 0.2827499985694885, 'stay_ratio': 0.306, 'boundary_hit_ratio': 0.322, 'revisit_ratio': 0.994}
DDQN primitive:  90%|███████████████████████████████▍   | 899/1000 [09:51<01:05,  1.55it/s, reward=-1.58, det=1.22, comp=0.56, best=0.29, eps=0.05]
[Best] episode=900 score=2.202 metrics={'eval_reward': -0.8975000000000012, 'eval_detected': 1.7, 'eval_completed': 0.7, 'eval_detected_value': 2.1, 'eval_completed_value': 0.9, 'eval_sensor_coverage': 0.26349999755620956, 'stay_ratio': 0.3273333333333333, 'boundary_hit_ratio': 0.2946666666666667, 'revisit_ratio': 0.994}

[Eval 900] {'eval_reward': -0.8975000000000012, 'eval_detected': 1.7, 'eval_completed': 0.7, 'eval_detected_value': 2.1, 'eval_completed_value': 0.9, 'eval_sensor_coverage': 0.26349999755620956, 'stay_ratio': 0.3273333333333333, 'boundary_hit_ratio': 0.2946666666666667, 'revisit_ratio': 0.994}
DDQN primitive:  95%|█████████████████████████████████▏ | 949/1000 [10:27<00:32,  1.58it/s, reward=-1.61, det=1.26, comp=0.36, best=2.20, eps=0.05]
[Eval 950] {'eval_reward': -2.9809999999999985, 'eval_detected': 1.2, 'eval_completed': 0.3, 'eval_detected_value': 1.5, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.210750000923872, 'stay_ratio': 0.39866666666666667, 'boundary_hit_ratio': 0.3446666666666667, 'revisit_ratio': 0.9946666666666667}
DDQN primitive: 100%|██████████████████████████████████▉| 999/1000 [11:02<00:00,  1.57it/s, reward=-2.91, det=1.02, comp=0.28, best=2.20, eps=0.05]
[Eval 1000] {'eval_reward': -0.8125000000000012, 'eval_detected': 1.1, 'eval_completed': 0.6, 'eval_detected_value': 1.3, 'eval_completed_value': 0.7, 'eval_sensor_coverage': 0.1767500028014183, 'stay_ratio': 0.6693333333333333, 'boundary_hit_ratio': 0.124, 'revisit_ratio': 0.994}
DDQN primitive: 100%|██████████████████████████████████| 1000/1000 [11:04<00:00,  1.50it/s, reward=-2.91, det=1.02, comp=0.28, best=2.20, eps=0.05]
Training complete.
Best checkpoint: runs/final/ddqn_seed42_1000/best.pt at episode 900, score=2.202
(.venv) kermit@kermit-PC-ART:~/Github/UAV-Search-Primitives_actions$ python scripts/evaluate.py --algo ddqn --checkpoint runs/final/ddqn_seed42_1000/best.pt --episodes 1000
python scripts/evaluate.py --algo ddqn --checkpoint runs/final/ddqn_seed43_1000/best.pt --episodes 1000
python scripts/evaluate.py --algo ddqn --checkpoint runs/final/ddqn_seed44_1000/best.pt --episodes 1000
reward_mean: -1.4918049999999987
reward_std: 3.1368396782390695
detected_mean: 1.204
completed_mean: 0.38
detected_value_mean: 1.524
completed_value_mean: 0.471
sensor_coverage_ratio_mean: 0.29155999977886676
stay_ratio: 0.4022133333333333
boundary_hit_ratio: 0.16311333333333333
revisit_ratio: 0.9937
action_counts: {'right': 22893, 'left': 24247, 'stay': 60332, 'down': 20814, 'up': 21714}
reward_mean: -1.35986
reward_std: 3.377811833184313
detected_mean: 1.346
completed_mean: 0.443
detected_value_mean: 1.68
completed_value_mean: 0.548
sensor_coverage_ratio_mean: 0.32698250002786516
stay_ratio: 0.17302
boundary_hit_ratio: 0.18209333333333333
revisit_ratio: 0.99336
action_counts: {'right': 39627, 'down': 25312, 'left': 45511, 'stay': 25953, 'up': 13597}
reward_mean: -2.208085
reward_std: 3.383496838741686
detected_mean: 1.128
completed_mean: 0.321
detected_value_mean: 1.416
completed_value_mean: 0.398
sensor_coverage_ratio_mean: 0.2787574996910989
stay_ratio: 0.46218952591841633
boundary_hit_ratio: 0.2251153856414908
revisit_ratio: 0.9935103913774245
action_counts: {'right': 21278, 'left': 21341, 'up': 15516, 'down': 22500, 'stay': 69297}
(.venv) kermit@kermit-PC-ART:~/Github/UAV-Search-Primitives_actions$ 

.venv) kermit@kermit-PC-ART:~/Github/UAV-Search-Primitives_actions$ python scripts/train.py \
  --algo ddqn \
  --episodes 1000 \
  --device cuda \
  --train-every 4 \
  --learning-starts 1000 \
  --eval-every 50 \
  --eval-episodes 10 \
  --seed 44 \
  --run-dir runs/final/ddqn_seed44_1000
Using device: cuda
torch threads: 1
train_every: 4, learning_starts: 1000
DDQN primitive:   5%|▊                | 49/1000 [00:22<08:43,  1.82it/s, reward=-0.18, det=1.32, comp=0.60, best=-1000000000000000000.00, eps=0.64]
[Best] episode=50 score=-6.540 metrics={'eval_reward': -7.340000000000006, 'eval_detected': 0.6, 'eval_completed': 0.1, 'eval_detected_value': 0.6, 'eval_completed_value': 0.1, 'eval_sensor_coverage': 0.1475000001490116, 'stay_ratio': 0.114, 'boundary_hit_ratio': 0.7333333333333333, 'revisit_ratio': 0.9933333333333333}

[Eval 50] {'eval_reward': -7.340000000000006, 'eval_detected': 0.6, 'eval_completed': 0.1, 'eval_detected_value': 0.6, 'eval_completed_value': 0.1, 'eval_sensor_coverage': 0.1475000001490116, 'stay_ratio': 0.114, 'boundary_hit_ratio': 0.7333333333333333, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  10%|███▍                               | 99/1000 [00:53<09:21,  1.60it/s, reward=-0.68, det=1.26, comp=0.66, best=-6.54, eps=0.29]
[Best] episode=100 score=-3.824 metrics={'eval_reward': -4.423999999999998, 'eval_detected': 0.4, 'eval_completed': 0.1, 'eval_detected_value': 0.5, 'eval_completed_value': 0.1, 'eval_sensor_coverage': 0.11550000011920929, 'stay_ratio': 0.4166666666666667, 'boundary_hit_ratio': 0.3446666666666667, 'revisit_ratio': 0.9946666666666667}

[Eval 100] {'eval_reward': -4.423999999999998, 'eval_detected': 0.4, 'eval_completed': 0.1, 'eval_detected_value': 0.5, 'eval_completed_value': 0.1, 'eval_sensor_coverage': 0.11550000011920929, 'stay_ratio': 0.4166666666666667, 'boundary_hit_ratio': 0.3446666666666667, 'revisit_ratio': 0.9946666666666667}
DDQN primitive:  15%|█████                             | 149/1000 [01:27<09:33,  1.48it/s, reward=-2.30, det=1.16, comp=0.36, best=-3.82, eps=0.05]
[Best] episode=150 score=-2.897 metrics={'eval_reward': -3.7969999999999993, 'eval_detected': 0.7, 'eval_completed': 0.1, 'eval_detected_value': 0.8, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.15825000032782555, 'stay_ratio': 0.2986666666666667, 'boundary_hit_ratio': 0.3566666666666667, 'revisit_ratio': 0.9973333333333333}

[Eval 150] {'eval_reward': -3.7969999999999993, 'eval_detected': 0.7, 'eval_completed': 0.1, 'eval_detected_value': 0.8, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.15825000032782555, 'stay_ratio': 0.2986666666666667, 'boundary_hit_ratio': 0.3566666666666667, 'revisit_ratio': 0.9973333333333333}
DDQN primitive:  20%|██████▊                           | 199/1000 [02:02<09:02,  1.48it/s, reward=-2.25, det=1.08, comp=0.40, best=-2.90, eps=0.05]
[Best] episode=200 score=-0.810 metrics={'eval_reward': -2.6095000000000006, 'eval_detected': 1.4, 'eval_completed': 0.2, 'eval_detected_value': 1.8, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.2244999997317791, 'stay_ratio': 0.3546666666666667, 'boundary_hit_ratio': 0.204, 'revisit_ratio': 0.9953333333333333}

[Eval 200] {'eval_reward': -2.6095000000000006, 'eval_detected': 1.4, 'eval_completed': 0.2, 'eval_detected_value': 1.8, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.2244999997317791, 'stay_ratio': 0.3546666666666667, 'boundary_hit_ratio': 0.204, 'revisit_ratio': 0.9953333333333333}
DDQN primitive:  25%|████████▍                         | 249/1000 [02:38<08:32,  1.47it/s, reward=-1.08, det=1.18, comp=0.42, best=-0.81, eps=0.05]
[Best] episode=250 score=-0.507 metrics={'eval_reward': -2.2065000000000023, 'eval_detected': 1.1, 'eval_completed': 0.3, 'eval_detected_value': 1.3, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.22274999916553498, 'stay_ratio': 0.4786666666666667, 'boundary_hit_ratio': 0.22866666666666666, 'revisit_ratio': 0.9953333333333333}

[Eval 250] {'eval_reward': -2.2065000000000023, 'eval_detected': 1.1, 'eval_completed': 0.3, 'eval_detected_value': 1.3, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.22274999916553498, 'stay_ratio': 0.4786666666666667, 'boundary_hit_ratio': 0.22866666666666666, 'revisit_ratio': 0.9953333333333333}
DDQN primitive:  30%|██████████▏                       | 299/1000 [03:13<07:53,  1.48it/s, reward=-1.02, det=1.34, comp=0.60, best=-0.51, eps=0.05]
[Eval 300] {'eval_reward': -5.357000000000008, 'eval_detected': 0.8, 'eval_completed': 0.0, 'eval_detected_value': 1.0, 'eval_completed_value': 0.0, 'eval_sensor_coverage': 0.21325000151991844, 'stay_ratio': 0.118, 'boundary_hit_ratio': 0.4713333333333333, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  35%|███████████▊                      | 349/1000 [03:48<07:22,  1.47it/s, reward=-2.46, det=0.98, comp=0.38, best=-0.51, eps=0.05]
[Eval 350] {'eval_reward': -2.3080000000000043, 'eval_detected': 1.3, 'eval_completed': 0.1, 'eval_detected_value': 1.6, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.1832499958574772, 'stay_ratio': 0.5493333333333333, 'boundary_hit_ratio': 0.204, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  40%|█████████████▌                    | 399/1000 [04:24<06:47,  1.48it/s, reward=-1.57, det=1.30, comp=0.56, best=-0.51, eps=0.05]
[Eval 400] {'eval_reward': -4.847999999999999, 'eval_detected': 1.0, 'eval_completed': 0.2, 'eval_detected_value': 1.3, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.20800000056624413, 'stay_ratio': 0.20466666666666666, 'boundary_hit_ratio': 0.47533333333333333, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  45%|███████████████▎                  | 449/1000 [04:59<06:10,  1.49it/s, reward=-4.38, det=0.98, comp=0.16, best=-0.51, eps=0.05]
[Best] episode=450 score=0.123 metrics={'eval_reward': -2.076999999999999, 'eval_detected': 1.2, 'eval_completed': 0.5, 'eval_detected_value': 1.5, 'eval_completed_value': 0.6, 'eval_sensor_coverage': 0.2640000030398369, 'stay_ratio': 0.36733333333333335, 'boundary_hit_ratio': 0.2806666666666667, 'revisit_ratio': 0.9933333333333333}

[Eval 450] {'eval_reward': -2.076999999999999, 'eval_detected': 1.2, 'eval_completed': 0.5, 'eval_detected_value': 1.5, 'eval_completed_value': 0.6, 'eval_sensor_coverage': 0.2640000030398369, 'stay_ratio': 0.36733333333333335, 'boundary_hit_ratio': 0.2806666666666667, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  50%|█████████████████▍                 | 499/1000 [05:34<05:37,  1.48it/s, reward=-3.01, det=1.28, comp=0.28, best=0.12, eps=0.05]
[Eval 500] {'eval_reward': -3.7855000000000047, 'eval_detected': 0.9, 'eval_completed': 0.3, 'eval_detected_value': 1.1, 'eval_completed_value': 0.3, 'eval_sensor_coverage': 0.25625000037252904, 'stay_ratio': 0.29533333333333334, 'boundary_hit_ratio': 0.348, 'revisit_ratio': 0.994}
DDQN primitive:  55%|███████████████████▏               | 549/1000 [06:09<05:05,  1.47it/s, reward=-2.83, det=1.02, comp=0.34, best=0.12, eps=0.05]
[Eval 550] {'eval_reward': -3.4940000000000024, 'eval_detected': 1.2, 'eval_completed': 0.3, 'eval_detected_value': 1.4, 'eval_completed_value': 0.3, 'eval_sensor_coverage': 0.29650000035762786, 'stay_ratio': 0.274, 'boundary_hit_ratio': 0.32133333333333336, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  60%|████████████████████▉              | 599/1000 [06:44<04:29,  1.49it/s, reward=-1.84, det=1.34, comp=0.46, best=0.12, eps=0.05]
[Eval 600] {'eval_reward': -3.7319999999999993, 'eval_detected': 1.1, 'eval_completed': 0.3, 'eval_detected_value': 1.5, 'eval_completed_value': 0.3, 'eval_sensor_coverage': 0.2862499989569187, 'stay_ratio': 0.38466666666666666, 'boundary_hit_ratio': 0.36733333333333335, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  65%|██████████████████████▋            | 649/1000 [07:19<03:58,  1.47it/s, reward=-1.29, det=1.28, comp=0.40, best=0.12, eps=0.05]
[Eval 650] {'eval_reward': -2.5459999999999985, 'eval_detected': 1.3, 'eval_completed': 0.2, 'eval_detected_value': 1.6, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.27699999660253527, 'stay_ratio': 0.42133333333333334, 'boundary_hit_ratio': 0.18, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  70%|████████████████████████▍          | 699/1000 [07:54<03:25,  1.46it/s, reward=-1.85, det=1.24, comp=0.38, best=0.12, eps=0.05]
[Best] episode=700 score=0.375 metrics={'eval_reward': -1.4250000000000012, 'eval_detected': 1.2, 'eval_completed': 0.3, 'eval_detected_value': 1.6, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.23599999770522118, 'stay_ratio': 0.658, 'boundary_hit_ratio': 0.14533333333333334, 'revisit_ratio': 0.9933333333333333}

[Eval 700] {'eval_reward': -1.4250000000000012, 'eval_detected': 1.2, 'eval_completed': 0.3, 'eval_detected_value': 1.6, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.23599999770522118, 'stay_ratio': 0.658, 'boundary_hit_ratio': 0.14533333333333334, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  75%|██████████████████████████▏        | 749/1000 [08:29<02:49,  1.48it/s, reward=-2.07, det=1.24, comp=0.32, best=0.37, eps=0.05]
[Eval 750] {'eval_reward': -2.959500000000001, 'eval_detected': 1.4, 'eval_completed': 0.2, 'eval_detected_value': 1.8, 'eval_completed_value': 0.3, 'eval_sensor_coverage': 0.2604999985545874, 'stay_ratio': 0.426, 'boundary_hit_ratio': 0.33066666666666666, 'revisit_ratio': 0.994}
DDQN primitive:  80%|███████████████████████████▉       | 799/1000 [09:04<02:15,  1.48it/s, reward=-2.36, det=1.08, comp=0.50, best=0.37, eps=0.05]
[Eval 800] {'eval_reward': -3.237999999999998, 'eval_detected': 1.1, 'eval_completed': 0.0, 'eval_detected_value': 1.3, 'eval_completed_value': 0.0, 'eval_sensor_coverage': 0.2650000013411045, 'stay_ratio': 0.368, 'boundary_hit_ratio': 0.196, 'revisit_ratio': 0.996}
DDQN primitive:  85%|█████████████████████████████▋     | 849/1000 [09:39<01:42,  1.47it/s, reward=-1.68, det=1.40, comp=0.40, best=0.37, eps=0.05]
[Eval 850] {'eval_reward': -3.8329999999999997, 'eval_detected': 0.8, 'eval_completed': 0.2, 'eval_detected_value': 0.9, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.22975000068545343, 'stay_ratio': 0.4646666666666667, 'boundary_hit_ratio': 0.31133333333333335, 'revisit_ratio': 0.996}
DDQN primitive:  90%|███████████████████████████████▍   | 899/1000 [10:14<01:08,  1.48it/s, reward=-2.40, det=1.08, comp=0.32, best=0.37, eps=0.05]
[Eval 900] {'eval_reward': -1.972, 'eval_detected': 1.1, 'eval_completed': 0.4, 'eval_detected_value': 1.2, 'eval_completed_value': 0.5, 'eval_sensor_coverage': 0.297249998152256, 'stay_ratio': 0.4246666666666667, 'boundary_hit_ratio': 0.21533333333333332, 'revisit_ratio': 0.9946666666666667}
DDQN primitive:  95%|█████████████████████████████████▏ | 949/1000 [10:42<00:23,  2.14it/s, reward=-1.63, det=1.28, comp=0.42, best=0.37, eps=0.05]
[Eval 950] {'eval_reward': -3.6639999999999993, 'eval_detected': 1.6, 'eval_completed': 0.4, 'eval_detected_value': 1.7, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.32374999970197677, 'stay_ratio': 0.20533333333333334, 'boundary_hit_ratio': 0.37933333333333336, 'revisit_ratio': 0.9933333333333333}
DDQN primitive: 100%|██████████████████████████████████▉| 999/1000 [10:59<00:00,  3.80it/s, reward=-2.02, det=1.20, comp=0.32, best=0.37, eps=0.05]
[Eval 1000] {'eval_reward': -3.138999999999998, 'eval_detected': 1.8, 'eval_completed': 0.3, 'eval_detected_value': 2.1, 'eval_completed_value': 0.3, 'eval_sensor_coverage': 0.30025000125169754, 'stay_ratio': 0.28933333333333333, 'boundary_hit_ratio': 0.30733333333333335, 'revisit_ratio': 0.9933333333333333}
DDQN primitive: 100%|██████████████████████████████████| 1000/1000 [11:00<00:00,  1.51it/s, reward=-2.02, det=1.20, comp=0.32, best=0.37, eps=0.05]
Training complete.
Best checkpoint: runs/final/ddqn_seed44_1000/best.pt at episode 700, score=0.375
(.venv) kermit@kermit-PC-ART:~/Github/UAV-Search-Primitives_actions$ 

(.venv) kermit@kermit-PC-ART:~/Github/UAV-Search-Primitives_actions$   python scripts/train.py \
  --algo ddqn \
  --episodes 1000 \
  --device cuda \
  --train-every 4 \
  --learning-starts 1000 \
  --eval-every 50 \
  --eval-episodes 10 \
  --seed 43 \
  --run-dir runs/final/ddqn_seed43_1000
Using device: cuda
torch threads: 1
train_every: 4, learning_starts: 1000
DDQN primitive:   5%|▉                 | 49/1000 [00:15<06:11,  2.56it/s, reward=0.15, det=1.18, comp=0.66, best=-1000000000000000000.00, eps=0.64]
[Best] episode=50 score=-0.828 metrics={'eval_reward': -2.9280000000000017, 'eval_detected': 1.3, 'eval_completed': 0.4, 'eval_detected_value': 1.6, 'eval_completed_value': 0.5, 'eval_sensor_coverage': 0.22600000202655793, 'stay_ratio': 0.4226666666666667, 'boundary_hit_ratio': 0.36866666666666664, 'revisit_ratio': 1.0}

[Eval 50] {'eval_reward': -2.9280000000000017, 'eval_detected': 1.3, 'eval_completed': 0.4, 'eval_detected_value': 1.6, 'eval_completed_value': 0.5, 'eval_sensor_coverage': 0.22600000202655793, 'stay_ratio': 0.4226666666666667, 'boundary_hit_ratio': 0.36866666666666664, 'revisit_ratio': 1.0}
DDQN primitive:  10%|███▍                               | 99/1000 [00:46<08:06,  1.85it/s, reward=-0.59, det=1.22, comp=0.68, best=-0.83, eps=0.29]
[Eval 100] {'eval_reward': -3.6850000000000023, 'eval_detected': 0.8, 'eval_completed': 0.5, 'eval_detected_value': 0.9, 'eval_completed_value': 0.5, 'eval_sensor_coverage': 0.2499999985098839, 'stay_ratio': 0.388, 'boundary_hit_ratio': 0.376, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  15%|█████                             | 149/1000 [01:20<07:39,  1.85it/s, reward=-2.61, det=1.12, comp=0.52, best=-0.83, eps=0.05]
[Eval 150] {'eval_reward': -2.711000000000001, 'eval_detected': 1.2, 'eval_completed': 0.3, 'eval_detected_value': 1.5, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.29049999862909315, 'stay_ratio': 0.41333333333333333, 'boundary_hit_ratio': 0.30866666666666664, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  20%|██████▊                           | 199/1000 [01:55<07:12,  1.85it/s, reward=-2.17, det=1.30, comp=0.44, best=-0.83, eps=0.05]
[Best] episode=200 score=-0.533 metrics={'eval_reward': -2.3330000000000073, 'eval_detected': 1.2, 'eval_completed': 0.3, 'eval_detected_value': 1.4, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.2459999993443489, 'stay_ratio': 0.334, 'boundary_hit_ratio': 0.24733333333333332, 'revisit_ratio': 0.9946666666666667}

[Eval 200] {'eval_reward': -2.3330000000000073, 'eval_detected': 1.2, 'eval_completed': 0.3, 'eval_detected_value': 1.4, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.2459999993443489, 'stay_ratio': 0.334, 'boundary_hit_ratio': 0.24733333333333332, 'revisit_ratio': 0.9946666666666667}
DDQN primitive:  25%|████████▍                         | 249/1000 [02:30<06:51,  1.82it/s, reward=-2.57, det=1.14, comp=0.26, best=-0.53, eps=0.05]
[Eval 250] {'eval_reward': -3.1095000000000015, 'eval_detected': 0.8, 'eval_completed': 0.3, 'eval_detected_value': 1.1, 'eval_completed_value': 0.5, 'eval_sensor_coverage': 0.2469999983906746, 'stay_ratio': 0.302, 'boundary_hit_ratio': 0.43, 'revisit_ratio': 0.994}
DDQN primitive:  30%|██████████▏                       | 299/1000 [03:05<06:29,  1.80it/s, reward=-3.31, det=1.06, comp=0.18, best=-0.53, eps=0.05]
[Best] episode=300 score=0.551 metrics={'eval_reward': -1.549000000000008, 'eval_detected': 1.1, 'eval_completed': 0.5, 'eval_detected_value': 1.5, 'eval_completed_value': 0.7, 'eval_sensor_coverage': 0.3522499993443489, 'stay_ratio': 0.10866666666666666, 'boundary_hit_ratio': 0.3, 'revisit_ratio': 0.9946666666666667}

[Eval 300] {'eval_reward': -1.549000000000008, 'eval_detected': 1.1, 'eval_completed': 0.5, 'eval_detected_value': 1.5, 'eval_completed_value': 0.7, 'eval_sensor_coverage': 0.3522499993443489, 'stay_ratio': 0.10866666666666666, 'boundary_hit_ratio': 0.3, 'revisit_ratio': 0.9946666666666667}
DDQN primitive:  35%|████████████▏                      | 349/1000 [03:40<06:10,  1.76it/s, reward=-1.35, det=1.26, comp=0.42, best=0.55, eps=0.05]
[Best] episode=350 score=0.551 metrics={'eval_reward': -1.3490000000000033, 'eval_detected': 1.5, 'eval_completed': 0.2, 'eval_detected_value': 1.9, 'eval_completed_value': 0.3, 'eval_sensor_coverage': 0.2872499987483025, 'stay_ratio': 0.554, 'boundary_hit_ratio': 0.12333333333333334, 'revisit_ratio': 0.9933333333333333}

[Eval 350] {'eval_reward': -1.3490000000000033, 'eval_detected': 1.5, 'eval_completed': 0.2, 'eval_detected_value': 1.9, 'eval_completed_value': 0.3, 'eval_sensor_coverage': 0.2872499987483025, 'stay_ratio': 0.554, 'boundary_hit_ratio': 0.12333333333333334, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  40%|█████████████▉                     | 399/1000 [04:15<05:46,  1.74it/s, reward=-2.12, det=1.22, comp=0.46, best=0.55, eps=0.05]
[Eval 400] {'eval_reward': -3.113000000000003, 'eval_detected': 1.1, 'eval_completed': 0.5, 'eval_detected_value': 1.2, 'eval_completed_value': 0.6, 'eval_sensor_coverage': 0.3002500027418137, 'stay_ratio': 0.112, 'boundary_hit_ratio': 0.3953333333333333, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  45%|███████████████▋                   | 449/1000 [04:50<05:30,  1.67it/s, reward=-2.28, det=1.10, comp=0.38, best=0.55, eps=0.05]
[Eval 450] {'eval_reward': -2.5180000000000007, 'eval_detected': 1.0, 'eval_completed': 0.4, 'eval_detected_value': 1.1, 'eval_completed_value': 0.5, 'eval_sensor_coverage': 0.23075000196695328, 'stay_ratio': 0.328, 'boundary_hit_ratio': 0.2833333333333333, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  50%|█████████████████▍                 | 499/1000 [05:25<05:14,  1.59it/s, reward=-2.64, det=1.10, comp=0.50, best=0.55, eps=0.05]
[Eval 500] {'eval_reward': -4.212000000000005, 'eval_detected': 0.9, 'eval_completed': 0.1, 'eval_detected_value': 1.0, 'eval_completed_value': 0.1, 'eval_sensor_coverage': 0.27124999463558197, 'stay_ratio': 0.27466666666666667, 'boundary_hit_ratio': 0.34, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  55%|███████████████████▏               | 549/1000 [06:00<04:57,  1.52it/s, reward=-2.42, det=1.12, comp=0.40, best=0.55, eps=0.05]
[Best] episode=550 score=2.047 metrics={'eval_reward': -0.6529999999999994, 'eval_detected': 1.7, 'eval_completed': 0.5, 'eval_detected_value': 2.0, 'eval_completed_value': 0.6, 'eval_sensor_coverage': 0.29700000286102296, 'stay_ratio': 0.37666666666666665, 'boundary_hit_ratio': 0.114, 'revisit_ratio': 0.9933333333333333}

[Eval 550] {'eval_reward': -0.6529999999999994, 'eval_detected': 1.7, 'eval_completed': 0.5, 'eval_detected_value': 2.0, 'eval_completed_value': 0.6, 'eval_sensor_coverage': 0.29700000286102296, 'stay_ratio': 0.37666666666666665, 'boundary_hit_ratio': 0.114, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  60%|████████████████████▉              | 599/1000 [06:35<04:25,  1.51it/s, reward=-1.82, det=1.02, comp=0.34, best=2.05, eps=0.05]
[Eval 600] {'eval_reward': -2.793499999999997, 'eval_detected': 1.8, 'eval_completed': 0.4, 'eval_detected_value': 2.1, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.26050000414252283, 'stay_ratio': 0.29333333333333333, 'boundary_hit_ratio': 0.29133333333333333, 'revisit_ratio': 0.994}
DDQN primitive:  65%|██████████████████████▋            | 649/1000 [07:10<03:51,  1.52it/s, reward=-2.61, det=0.96, comp=0.14, best=2.05, eps=0.05]
[Eval 650] {'eval_reward': -3.5810000000000004, 'eval_detected': 1.3, 'eval_completed': 0.1, 'eval_detected_value': 1.4, 'eval_completed_value': 0.1, 'eval_sensor_coverage': 0.204500000923872, 'stay_ratio': 0.5346666666666666, 'boundary_hit_ratio': 0.27266666666666667, 'revisit_ratio': 0.9946666666666667}
DDQN primitive:  70%|████████████████████████▍          | 699/1000 [07:45<03:20,  1.50it/s, reward=-2.55, det=0.98, comp=0.30, best=2.05, eps=0.05]
[Eval 700] {'eval_reward': -2.3379999999999987, 'eval_detected': 1.9, 'eval_completed': 0.6, 'eval_detected_value': 2.2, 'eval_completed_value': 0.7, 'eval_sensor_coverage': 0.2930000007152557, 'stay_ratio': 0.23533333333333334, 'boundary_hit_ratio': 0.37133333333333335, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  75%|██████████████████████████▏        | 749/1000 [08:20<02:47,  1.50it/s, reward=-1.05, det=1.16, comp=0.46, best=2.05, eps=0.05]
[Eval 750] {'eval_reward': -1.123999999999996, 'eval_detected': 1.6, 'eval_completed': 0.6, 'eval_detected_value': 2.0, 'eval_completed_value': 0.7, 'eval_sensor_coverage': 0.24350000098347663, 'stay_ratio': 0.4186666666666667, 'boundary_hit_ratio': 0.206, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  80%|███████████████████████████▉       | 799/1000 [08:55<02:13,  1.51it/s, reward=-2.29, det=1.20, comp=0.32, best=2.05, eps=0.05]
[Best] episode=800 score=2.886 metrics={'eval_reward': -0.41400000000000003, 'eval_detected': 1.9, 'eval_completed': 0.7, 'eval_detected_value': 2.2, 'eval_completed_value': 0.8, 'eval_sensor_coverage': 0.35950000286102296, 'stay_ratio': 0.2806666666666667, 'boundary_hit_ratio': 0.14466666666666667, 'revisit_ratio': 0.9933333333333333}

[Eval 800] {'eval_reward': -0.41400000000000003, 'eval_detected': 1.9, 'eval_completed': 0.7, 'eval_detected_value': 2.2, 'eval_completed_value': 0.8, 'eval_sensor_coverage': 0.35950000286102296, 'stay_ratio': 0.2806666666666667, 'boundary_hit_ratio': 0.14466666666666667, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  85%|█████████████████████████████▋     | 849/1000 [09:31<01:40,  1.50it/s, reward=-0.80, det=1.60, comp=0.56, best=2.89, eps=0.05]
[Eval 850] {'eval_reward': -1.1510000000000005, 'eval_detected': 1.5, 'eval_completed': 0.5, 'eval_detected_value': 1.6, 'eval_completed_value': 0.6, 'eval_sensor_coverage': 0.3002499997615814, 'stay_ratio': 0.47, 'boundary_hit_ratio': 0.14933333333333335, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  90%|███████████████████████████████▍   | 899/1000 [10:06<01:07,  1.50it/s, reward=-0.80, det=1.46, comp=0.40, best=2.89, eps=0.05]
[Best] episode=900 score=3.953 metrics={'eval_reward': 0.45299999999999485, 'eval_detected': 1.9, 'eval_completed': 0.8, 'eval_detected_value': 2.3, 'eval_completed_value': 1.1, 'eval_sensor_coverage': 0.35024999678134916, 'stay_ratio': 0.19266666666666668, 'boundary_hit_ratio': 0.23066666666666666, 'revisit_ratio': 0.9933333333333333}

[Eval 900] {'eval_reward': 0.45299999999999485, 'eval_detected': 1.9, 'eval_completed': 0.8, 'eval_detected_value': 2.3, 'eval_completed_value': 1.1, 'eval_sensor_coverage': 0.35024999678134916, 'stay_ratio': 0.19266666666666668, 'boundary_hit_ratio': 0.23066666666666666, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  95%|█████████████████████████████████▏ | 949/1000 [10:41<00:33,  1.51it/s, reward=-1.28, det=1.58, comp=0.44, best=3.95, eps=0.05]
[Eval 950] {'eval_reward': -0.590000000000007, 'eval_detected': 1.6, 'eval_completed': 0.6, 'eval_detected_value': 1.9, 'eval_completed_value': 0.7, 'eval_sensor_coverage': 0.3259999960660934, 'stay_ratio': 0.32666666666666666, 'boundary_hit_ratio': 0.12866666666666668, 'revisit_ratio': 0.9933333333333333}
DDQN primitive: 100%|██████████████████████████████████▉| 999/1000 [11:06<00:00,  2.20it/s, reward=-0.78, det=1.36, comp=0.46, best=3.95, eps=0.05]
[Eval 1000] {'eval_reward': -0.06299999999999484, 'eval_detected': 1.7, 'eval_completed': 0.4, 'eval_detected_value': 2.1, 'eval_completed_value': 0.6, 'eval_sensor_coverage': 0.3172499999403954, 'stay_ratio': 0.5473333333333333, 'boundary_hit_ratio': 0.098, 'revisit_ratio': 0.9933333333333333}
DDQN primitive: 100%|██████████████████████████████████| 1000/1000 [11:08<00:00,  1.50it/s, reward=-0.78, det=1.36, comp=0.46, best=3.95, eps=0.05]
Training complete.
Best checkpoint: runs/final/ddqn_seed43_1000/best.pt at episode 900, score=3.953
(.venv) kermit@kermit-PC-ART:~/Github/UAV-Search-Primitives_actions$ 