@echo off

echo Creating Python environment...
call "C:\Users\%USERNAME%\miniconda3\Scripts\conda" create -n openvino_qwen python=3.10 -y

echo Activating environment...
call "C:\Users\%USERNAME%\miniconda3\Scripts\activate.bat" openvino_qwen

echo Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

echo Checking installed packages...
conda list


echo Installation complete!
pause
