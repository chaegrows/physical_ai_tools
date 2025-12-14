# Physical AI Tools

This repository offers an interface for developing physical AI applications using LeRobot and ROS 2. For detailed usage instructions, please refer to the documentation below.
  - [Documentation for AI Worker](https://ai.robotis.com/)

To clone this repository:
```bash
git clone -b jazzy https://github.com/ROBOTIS-GIT/physical_ai_tools.git --recursive
```

To learn more about the ROS 2 packages for the AI Worker, visit:
  - [AI Worker ROS 2 Packages](https://github.com/ROBOTIS-GIT/ai_worker)

To explore our open-source platforms in a simulation environment, visit:
  - [Simulation Models](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie)

For usage instructions and demonstrations of the AI Worker, check out:
  - [Tutorial Videos](https://www.youtube.com/@ROBOTISOpenSourceTeam)

To access datasets and pre-trained models for our open-source platforms, see:
  - [AI Models & Datasets](https://huggingface.co/ROBOTIS)

To use the Docker image for running ROS packages and Physical AI tools with the AI Worker, visit:
  - [Docker Images](https://hub.docker.com/r/robotis/ros/tags)


## docker rebuild
```bash
# Docker 컨테이너 밖에서 (호스트 시스템)
cd /media/metafarmers/7440017c-39b5-476a-a24b-4fb0a9c24140/chae/dualArm_rosbag/physical_ai_tools

# Manager 재빌드
docker compose build physical_ai_manager

# 또는 전체 재시작
docker compose down
docker compose up -d
```