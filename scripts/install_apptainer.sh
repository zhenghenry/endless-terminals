# 1) Update & add the Apptainer PPA
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update

# 2) Install (rootless by default)
# sudo apt install -y apptainer
# If you specifically want setuid mode, install this package instead:
# sudo apt install -y apptainer-suid
sudo apt install -y apptainer-suid

sudo mount -o remount,hidepid=0 /proc