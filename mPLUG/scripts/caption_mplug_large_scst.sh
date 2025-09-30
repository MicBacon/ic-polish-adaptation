pip install colorlog
apt-get update
apt-get install default-jdk
apt-get install default-jre

#python -m torch.distributed.launch --nproc_per_node=4 --master_port=3224  --use_env caption_mplug_scst.py \
    python caption_mplug_scst.py \
    --config ./configs/caption_mplug_large_scst.yaml \
    --output_dir output/coco_caption_large_scst \
    --checkpoint ./checkpoint_09.pth \
    --text_encoder bert-base-uncased \
    --text_decoder bert-base-uncased \
    --do_two_optim \
    --min_length 8 \
    --max_length 25 \
    --max_input_length 25
