Seeds to do :
tests:
seed 42:

python scripts/train.py \
  --algo ddqn \
  --episodes 1000 \
  --device cuda \
  --train-every 4 \
  --learning-starts 1000 \
  --eval-every 50 \
  --eval-episodes 10 \
  --seed 42 \
  --run-dir runs/final/ddqn_seed42_1000

  seed 43:

  python scripts/train.py \
  --algo ddqn \
  --episodes 1000 \
  --device cuda \
  --train-every 4 \
  --learning-starts 1000 \
  --eval-every 50 \
  --eval-episodes 10 \
  --seed 43 \
  --run-dir runs/final/ddqn_seed43_1000

  seed 44:
  python scripts/train.py \
  --algo ddqn \
  --episodes 1000 \
  --device cuda \
  --train-every 4 \
  --learning-starts 1000 \
  --eval-every 50 \
  --eval-episodes 10 \
  --seed 44 \
  --run-dir runs/final/ddqn_seed44_1000

python scripts/evaluate.py --algo ddqn --checkpoint runs/final/ddqn_seed42_1000/best.pt --episodes 1000
python scripts/evaluate.py --algo ddqn --checkpoint runs/final/ddqn_seed43_1000/best.pt --episodes 1000
python scripts/evaluate.py --algo ddqn --checkpoint runs/final/ddqn_seed44_1000/best.pt --episodes 1000

BDQN :

python scripts/train.py \
  --algo bdqn \
  --episodes 1000 \
  --device cuda \
  --train-every 4 \
  --learning-starts 1000 \
  --posterior-update-period 500 \
  --eval-every 50 \
  --eval-episodes 10 \
  --seed 42 \
  --run-dir runs/final/bdqn_seed42_1000
  
python scripts/train.py --algo bdqn --episodes 1000 --device cuda --train-every 4 --learning-starts 1000 --posterior-update-period 500 --eval-every 50 --eval-episodes 10 --seed 43 --run-dir runs/final/bdqn_seed43_1000
python scripts/train.py --algo bdqn --episodes 1000 --device cuda --train-every 4 --learning-starts 1000 --posterior-update-period 500 --eval-every 50 --eval-episodes 10 --seed 44 --run-dir runs/final/bdqn_seed44_1000

python scripts/evaluate.py --algo bdqn --checkpoint runs/final/bdqn_seed42_1000/best.pt --episodes 1000
python scripts/evaluate.py --algo bdqn --checkpoint runs/final/bdqn_seed43_1000/best.pt --episodes 1000
python scripts/evaluate.py --algo bdqn --checkpoint runs/final/bdqn_seed44_1000/best.pt --episodes 1000