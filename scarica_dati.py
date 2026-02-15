from roboflow import Roboflow
rf = Roboflow(api_key="aKSA56Lsw2zKyWAmV7QJ")
project = rf.workspace("pomodori-gqsky").project("pomodori")
version = project.version(1)
dataset = version.download("yolov11")