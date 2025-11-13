# Terminal 1
ssh robotis@ffw-SNPR48A1027.local
cd ai_worker

 ./docker/container.sh start

# Enter running container
./docker/container.sh enter

ffw_sg2_ai

ffw_sg2_follower_ai

# Terminal 2
cd chae/physical_ai_server
./docker/contatiner.sh start
./docker/container.sh enter
ai_server



del

3-2
3-4 확인
4-4 확인
5-1 버리기
7-2
 
 채영
 2-4 한손으로 땀
 3 전체 확인 필요
 4-4 지우라
 5-3
 6-5-
 8-3
 8-4
 8-5
 
 8'-3
 9-5
 10-2 
 
 
 python -m lerobot.scripts.train \
  --dataset.root=/root/ros2_ws/src/physical_ai_tools/docker/huggingface/lerobot/251108_all/all_data \
  --policy.type=pi0fast \
  --output_dir=outputs/train/251108_pi0fast\
  --policy.device=cuda \
  --log_freq=100 \
  --save_freq=1000 \
  --num_workers=1 \
  --batch_size=1 \ 
  --policy.push_to_hub=false
