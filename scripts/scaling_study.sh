#for node in 1 2 4 8 16 32 64 128 256;do
#for node in 256;do
#for node in 32;do
for node in 64;do
    #python run.py train --nodes=$node --dataset=imagenet1k --template=vitb8 --batch-size-per-gpu=8  --epochs=1 --warmup-epochs=0 --warmup-teacher-temp-epochs=0 --folder=results/scaling/vitb8/$node  --num-batches 100
    python run.py train --nodes=$node --dataset=laion400m --template=vitb8 --batch-size-per-gpu=8  --epochs=1 --warmup-epochs=0 --warmup-teacher-temp-epochs=0 --folder=results/scaling2/vitb8/$node  --num-batches 100
    #python run.py train --nodes=$node --dataset=imagenet1k --template=vitb8 --batch-size-per-gpu=8  --epochs=1 --warmup-epochs=0 --warmup-teacher-temp-epochs=0 --folder=results/scaling3/vitb8/$node  --num-batches 100 --local-crops-number=5
done
