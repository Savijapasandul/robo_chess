#READ ME#
![OS](https://img.shields.io/ubuntu/v/ubuntu-wallpapers/noble)
![ROS_2](https://img.shields.io/ros/v/jazzy/rclcpp)

To install wsl & ubuntu 24.04 LTS:
  - wsl --install -d Ubuntu-24.04
  - wsl --set-default Ubuntu-24.04

To install venv in ubuntu/home:
  - sudo apt update
  - sudo apt upgrade
  - sudo apt install python3-pip 
  - sudo apt install python3-venv 
  - python3 -m venv venv
  - echo 'source ~/venv/bin/activate' >> ~/.bashrc
  - source ~/.bashrc



Please backup (using the steps I mention in my previous answer) before trying this.

wsl --shutdown (from PowerShell or CMD)
In Windows, run the Registry Editor
Find \HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Lxss
Find the key in there that has DistributionName of Ubuntu20.04LTS. Change the Ubuntu20.04LTS to Ubuntu-20.04.
In theory, that may fix the problem by changing the distribution name back to what it should be.
To install ros2 Jazzy:
- https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
- echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
- echo "export ROS_DOMAIN_ID=12" >> ~/.bashrc
- printenv | grep -i ROS
- echo "source install/setup.bash" >> ~/.bashrc

