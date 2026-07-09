Here are all the evals stats from the morning pull:
(.venv) kermit@kermit-PC-ART:~/Github/UAV-Search-Primitives_actions$   python scripts/train.py \
  --algo ddqn \
  --episodes 1000 \
  --device cuda \
  --train-every 4 \
  --learning-starts 1000 \
  --eval-every 50 \
  --eval-episodes 10 \
  --seed 44 \
  --run-dir runs/fast/ddqn_seed0_1000
Using device: cuda
torch threads: 1
train_every: 4, learning_starts: 1000
DDQN primitive:   5%|▊                | 49/1000 [00:22<08:29,  1.87it/s, reward=-0.18, det=1.32, comp=0.60, best=-1000000000000000000.00, eps=0.64]^DDQN primitive:   5%|▊                | 49/1000 [00:24<07:56,  2.00it/s, reward=-0.18, det=1.32, comp=0.60, best=-1000000000000000000.00, eps=0.64]
Traceback (most recent call last):
  File "/home/kermit/Github/UAV-Search-Primitives_actions/scripts/train.py", line 330, in <module>
    main()
  File "/home/kermit/Github/UAV-Search-Primitives_actions/scripts/train.py", line 273, in main
    metrics = evaluate(agent, args, episodes=args.eval_episodes)
  File "/home/kermit/Github/UAV-Search-Primitives_actions/scripts/train.py", line 90, in evaluate
    action = agent.act(obs, explore=False, action_mask=env.action_mask())
  File "/home/kermit/Github/UAV-Search-Primitives_actions/.venv/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/home/kermit/Github/UAV-Search-Primitives_actions/.venv/lib/python3.10/site-packages/uav_search_belief20/agents/dqn_agent.py", line 65, in act
    x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
KeyboardInterrupt

(.venv) kermit@kermit-PC-ART:~/Github/UAV-Search-Primitives_actions$   python scripts/train.py \
  --algo ddqn \
  --episodes 1000 \
  --device cuda \
  --train-every 4 \
  --learning-starts 1000 \
  --eval-every 50 \
  --eval-episodes 10 \
  --seed 44 \
  --run-dir runs/fast/ddqn_seed44_1000
Using device: cuda
torch threads: 1
train_every: 4, learning_starts: 1000
DDQN primitive:   5%|▊                | 49/1000 [00:20<07:59,  1.98it/s, reward=-0.18, det=1.32, comp=0.60, best=-1000000000000000000.00, eps=0.64]
[Best] episode=50 score=-6.540 metrics={'eval_reward': -7.340000000000006, 'eval_detected': 0.6, 'eval_completed': 0.1, 'eval_detected_value': 0.6, 'eval_completed_value': 0.1, 'eval_sensor_coverage': 0.1475000001490116, 'stay_ratio': 0.114, 'boundary_hit_ratio': 0.7333333333333333, 'revisit_ratio': 0.9933333333333333}

[Eval 50] {'eval_reward': -7.340000000000006, 'eval_detected': 0.6, 'eval_completed': 0.1, 'eval_detected_value': 0.6, 'eval_completed_value': 0.1, 'eval_sensor_coverage': 0.1475000001490116, 'stay_ratio': 0.114, 'boundary_hit_ratio': 0.7333333333333333, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  10%|███▍                               | 99/1000 [00:44<08:26,  1.78it/s, reward=-0.68, det=1.26, comp=0.66, best=-6.54, eps=0.29]
[Best] episode=100 score=-3.824 metrics={'eval_reward': -4.423999999999998, 'eval_detected': 0.4, 'eval_completed': 0.1, 'eval_detected_value': 0.5, 'eval_completed_value': 0.1, 'eval_sensor_coverage': 0.11550000011920929, 'stay_ratio': 0.4166666666666667, 'boundary_hit_ratio': 0.3446666666666667, 'revisit_ratio': 0.9946666666666667}

[Eval 100] {'eval_reward': -4.423999999999998, 'eval_detected': 0.4, 'eval_completed': 0.1, 'eval_detected_value': 0.5, 'eval_completed_value': 0.1, 'eval_sensor_coverage': 0.11550000011920929, 'stay_ratio': 0.4166666666666667, 'boundary_hit_ratio': 0.3446666666666667, 'revisit_ratio': 0.9946666666666667}
DDQN primitive:  15%|█████                             | 149/1000 [01:21<10:03,  1.41it/s, reward=-2.30, det=1.16, comp=0.36, best=-3.82, eps=0.05]
[Best] episode=150 score=-2.897 metrics={'eval_reward': -3.7969999999999993, 'eval_detected': 0.7, 'eval_completed': 0.1, 'eval_detected_value': 0.8, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.15825000032782555, 'stay_ratio': 0.2986666666666667, 'boundary_hit_ratio': 0.3566666666666667, 'revisit_ratio': 0.9973333333333333}

[Eval 150] {'eval_reward': -3.7969999999999993, 'eval_detected': 0.7, 'eval_completed': 0.1, 'eval_detected_value': 0.8, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.15825000032782555, 'stay_ratio': 0.2986666666666667, 'boundary_hit_ratio': 0.3566666666666667, 'revisit_ratio': 0.9973333333333333}
DDQN primitive:  20%|██████▊                           | 199/1000 [01:57<09:04,  1.47it/s, reward=-2.25, det=1.08, comp=0.40, best=-2.90, eps=0.05]
[Best] episode=200 score=-0.810 metrics={'eval_reward': -2.6095000000000006, 'eval_detected': 1.4, 'eval_completed': 0.2, 'eval_detected_value': 1.8, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.2244999997317791, 'stay_ratio': 0.3546666666666667, 'boundary_hit_ratio': 0.204, 'revisit_ratio': 0.9953333333333333}

[Eval 200] {'eval_reward': -2.6095000000000006, 'eval_detected': 1.4, 'eval_completed': 0.2, 'eval_detected_value': 1.8, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.2244999997317791, 'stay_ratio': 0.3546666666666667, 'boundary_hit_ratio': 0.204, 'revisit_ratio': 0.9953333333333333}
DDQN primitive:  25%|████████▍                         | 249/1000 [02:32<08:26,  1.48it/s, reward=-1.08, det=1.18, comp=0.42, best=-0.81, eps=0.05]
[Best] episode=250 score=-0.507 metrics={'eval_reward': -2.2065000000000023, 'eval_detected': 1.1, 'eval_completed': 0.3, 'eval_detected_value': 1.3, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.22274999916553498, 'stay_ratio': 0.4786666666666667, 'boundary_hit_ratio': 0.22866666666666666, 'revisit_ratio': 0.9953333333333333}

[Eval 250] {'eval_reward': -2.2065000000000023, 'eval_detected': 1.1, 'eval_completed': 0.3, 'eval_detected_value': 1.3, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.22274999916553498, 'stay_ratio': 0.4786666666666667, 'boundary_hit_ratio': 0.22866666666666666, 'revisit_ratio': 0.9953333333333333}
DDQN primitive:  30%|██████████▏                       | 299/1000 [03:07<07:39,  1.53it/s, reward=-1.02, det=1.34, comp=0.60, best=-0.51, eps=0.05]
[Eval 300] {'eval_reward': -5.357000000000008, 'eval_detected': 0.8, 'eval_completed': 0.0, 'eval_detected_value': 1.0, 'eval_completed_value': 0.0, 'eval_sensor_coverage': 0.21325000151991844, 'stay_ratio': 0.118, 'boundary_hit_ratio': 0.4713333333333333, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  35%|███████████▊                      | 349/1000 [03:42<06:51,  1.58it/s, reward=-2.46, det=0.98, comp=0.38, best=-0.51, eps=0.05]
[Eval 350] {'eval_reward': -2.3080000000000043, 'eval_detected': 1.3, 'eval_completed': 0.1, 'eval_detected_value': 1.6, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.1832499958574772, 'stay_ratio': 0.5493333333333333, 'boundary_hit_ratio': 0.204, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  40%|█████████████▌                    | 399/1000 [04:17<06:18,  1.59it/s, reward=-1.57, det=1.30, comp=0.56, best=-0.51, eps=0.05]
[Eval 400] {'eval_reward': -4.847999999999999, 'eval_detected': 1.0, 'eval_completed': 0.2, 'eval_detected_value': 1.3, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.20800000056624413, 'stay_ratio': 0.20466666666666666, 'boundary_hit_ratio': 0.47533333333333333, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  45%|███████████████▎                  | 449/1000 [04:52<05:49,  1.58it/s, reward=-4.38, det=0.98, comp=0.16, best=-0.51, eps=0.05]
[Best] episode=450 score=0.123 metrics={'eval_reward': -2.076999999999999, 'eval_detected': 1.2, 'eval_completed': 0.5, 'eval_detected_value': 1.5, 'eval_completed_value': 0.6, 'eval_sensor_coverage': 0.2640000030398369, 'stay_ratio': 0.36733333333333335, 'boundary_hit_ratio': 0.2806666666666667, 'revisit_ratio': 0.9933333333333333}

[Eval 450] {'eval_reward': -2.076999999999999, 'eval_detected': 1.2, 'eval_completed': 0.5, 'eval_detected_value': 1.5, 'eval_completed_value': 0.6, 'eval_sensor_coverage': 0.2640000030398369, 'stay_ratio': 0.36733333333333335, 'boundary_hit_ratio': 0.2806666666666667, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  50%|█████████████████▍                 | 499/1000 [05:27<05:23,  1.55it/s, reward=-3.01, det=1.28, comp=0.28, best=0.12, eps=0.05]
[Eval 500] {'eval_reward': -3.7855000000000047, 'eval_detected': 0.9, 'eval_completed': 0.3, 'eval_detected_value': 1.1, 'eval_completed_value': 0.3, 'eval_sensor_coverage': 0.25625000037252904, 'stay_ratio': 0.29533333333333334, 'boundary_hit_ratio': 0.348, 'revisit_ratio': 0.994}
DDQN primitive:  55%|███████████████████▏               | 549/1000 [06:02<04:51,  1.55it/s, reward=-2.83, det=1.02, comp=0.34, best=0.12, eps=0.05]
[Eval 550] {'eval_reward': -3.4940000000000024, 'eval_detected': 1.2, 'eval_completed': 0.3, 'eval_detected_value': 1.4, 'eval_completed_value': 0.3, 'eval_sensor_coverage': 0.29650000035762786, 'stay_ratio': 0.274, 'boundary_hit_ratio': 0.32133333333333336, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  60%|████████████████████▉              | 599/1000 [06:37<04:19,  1.55it/s, reward=-1.84, det=1.34, comp=0.46, best=0.12, eps=0.05]
[Eval 600] {'eval_reward': -3.7319999999999993, 'eval_detected': 1.1, 'eval_completed': 0.3, 'eval_detected_value': 1.5, 'eval_completed_value': 0.3, 'eval_sensor_coverage': 0.2862499989569187, 'stay_ratio': 0.38466666666666666, 'boundary_hit_ratio': 0.36733333333333335, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  65%|██████████████████████▋            | 649/1000 [07:12<03:48,  1.54it/s, reward=-1.29, det=1.28, comp=0.40, best=0.12, eps=0.05]
[Eval 650] {'eval_reward': -2.5459999999999985, 'eval_detected': 1.3, 'eval_completed': 0.2, 'eval_detected_value': 1.6, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.27699999660253527, 'stay_ratio': 0.42133333333333334, 'boundary_hit_ratio': 0.18, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  70%|████████████████████████▍          | 699/1000 [07:47<03:15,  1.54it/s, reward=-1.85, det=1.24, comp=0.38, best=0.12, eps=0.05]
[Best] episode=700 score=0.375 metrics={'eval_reward': -1.4250000000000012, 'eval_detected': 1.2, 'eval_completed': 0.3, 'eval_detected_value': 1.6, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.23599999770522118, 'stay_ratio': 0.658, 'boundary_hit_ratio': 0.14533333333333334, 'revisit_ratio': 0.9933333333333333}

[Eval 700] {'eval_reward': -1.4250000000000012, 'eval_detected': 1.2, 'eval_completed': 0.3, 'eval_detected_value': 1.6, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.23599999770522118, 'stay_ratio': 0.658, 'boundary_hit_ratio': 0.14533333333333334, 'revisit_ratio': 0.9933333333333333}
DDQN primitive:  75%|██████████████████████████▏        | 749/1000 [08:22<02:45,  1.52it/s, reward=-2.07, det=1.24, comp=0.32, best=0.37, eps=0.05]
[Eval 750] {'eval_reward': -2.959500000000001, 'eval_detected': 1.4, 'eval_completed': 0.2, 'eval_detected_value': 1.8, 'eval_completed_value': 0.3, 'eval_sensor_coverage': 0.2604999985545874, 'stay_ratio': 0.426, 'boundary_hit_ratio': 0.33066666666666666, 'revisit_ratio': 0.994}
DDQN primitive:  80%|███████████████████████████▉       | 799/1000 [08:57<02:12,  1.51it/s, reward=-2.36, det=1.08, comp=0.50, best=0.37, eps=0.05]
[Eval 800] {'eval_reward': -3.237999999999998, 'eval_detected': 1.1, 'eval_completed': 0.0, 'eval_detected_value': 1.3, 'eval_completed_value': 0.0, 'eval_sensor_coverage': 0.2650000013411045, 'stay_ratio': 0.368, 'boundary_hit_ratio': 0.196, 'revisit_ratio': 0.996}
DDQN primitive:  85%|█████████████████████████████▋     | 849/1000 [09:31<01:41,  1.49it/s, reward=-1.68, det=1.40, comp=0.40, best=0.37, eps=0.05]
[Eval 850] {'eval_reward': -3.8329999999999997, 'eval_detected': 0.8, 'eval_completed': 0.2, 'eval_detected_value': 0.9, 'eval_completed_value': 0.2, 'eval_sensor_coverage': 0.22975000068545343, 'stay_ratio': 0.4646666666666667, 'boundary_hit_ratio': 0.31133333333333335, 'revisit_ratio': 0.996}
DDQN primitive:  90%|███████████████████████████████▍   | 899/1000 [10:06<01:06,  1.51it/s, reward=-2.40, det=1.08, comp=0.32, best=0.37, eps=0.05]
[Eval 900] {'eval_reward': -1.972, 'eval_detected': 1.1, 'eval_completed': 0.4, 'eval_detected_value': 1.2, 'eval_completed_value': 0.5, 'eval_sensor_coverage': 0.297249998152256, 'stay_ratio': 0.4246666666666667, 'boundary_hit_ratio': 0.21533333333333332, 'revisit_ratio': 0.9946666666666667}
DDQN primitive:  95%|█████████████████████████████████▏ | 949/1000 [10:31<00:23,  2.16it/s, reward=-1.63, det=1.28, comp=0.42, best=0.37, eps=0.05]
[Eval 950] {'eval_reward': -3.6639999999999993, 'eval_detected': 1.6, 'eval_completed': 0.4, 'eval_detected_value': 1.7, 'eval_completed_value': 0.4, 'eval_sensor_coverage': 0.32374999970197677, 'stay_ratio': 0.20533333333333334, 'boundary_hit_ratio': 0.37933333333333336, 'revisit_ratio': 0.9933333333333333}
DDQN primitive: 100%|██████████████████████████████████▉| 999/1000 [10:55<00:00,  2.17it/s, reward=-2.02, det=1.20, comp=0.32, best=0.37, eps=0.05]
[Eval 1000] {'eval_reward': -3.138999999999998, 'eval_detected': 1.8, 'eval_completed': 0.3, 'eval_detected_value': 2.1, 'eval_completed_value': 0.3, 'eval_sensor_coverage': 0.30025000125169754, 'stay_ratio': 0.28933333333333333, 'boundary_hit_ratio': 0.30733333333333335, 'revisit_ratio': 0.9933333333333333}
DDQN primitive: 100%|██████████████████████████████████| 1000/1000 [10:57<00:00,  1.52it/s, reward=-2.02, det=1.20, comp=0.32, best=0.37, eps=0.05]
Training complete.
Best checkpoint: runs/fast/ddqn_seed44_1000/best.pt at episode 700, score=0.375
(.venv) kermit@kermit-PC-ART:~/Github/UAV-Search-Primitives_actions$  python scripts/evaluate.py \
  --algo ddqn \
  --checkpoint runs/fast/ddqn_seed44_1000/best.pt
  --episodes 1000 \
 python scripts/evaluate.py \
  --algo ddqn \
  --checkpoint runs/fast/ddqn_seed43_1000/best.pt
  --episodes 1000 \
   python scripts/evaluate.py \
  --algo ddqn \
  --checkpoint runs/fast/ddqn_seed42_1000/best.pt
  --episodes 1000 \
reward_mean: -2.272725
reward_std: 3.3312951609509147
detected_mean: 1.125
completed_mean: 0.27
detected_value_mean: 1.415
completed_value_mean: 0.34
sensor_coverage_ratio_mean: 0.2798374992236495
stay_ratio: 0.4819333333333333
boundary_hit_ratio: 0.21316666666666667
revisit_ratio: 0.9935
action_counts: {'right': 3988, 'left': 3956, 'up': 3292, 'down': 4306, 'stay': 14458}
--episodes: command not found
--episodes: command not found
> 
--episodes: command not found