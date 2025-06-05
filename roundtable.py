# from controller import Supervisor
# 
# robot = Supervisor()
# time_step = int(robot.getBasicTimeStep())
# 
# # 모터 디바이스 획득
# motor1 = robot.getDevice('motor 1')
# motor2 = robot.getDevice('motor 2')
# motor3 = robot.getDevice('motor 3')
# 
# # Supervisor로부터 DEF가 "Ball"인 노드 획득
# ball_node = robot.getFromDef('Ball')
# translation_field = ball_node.getField("translation")
# 
# # 위치 제어 모드 활성화
# motor1.setPosition(0.0)
# motor2.setPosition(0.0)
# motor3.setPosition(0.0)
# 
# counter = 0
# while robot.step(time_step) != -1:
#     ball_pos = translation_field.getSFVec3f()
#     print(f"ball x: {ball_pos[0]}, ball y = {ball_pos[1]}, ball z = {ball_pos[2]}")
#     if ball_pos[2] < 0.09:
#         robot.simulationReset()    
#     counter += 1
#     if counter == 50:
#         motor1.setPosition(-0.1) # red 축
#     elif counter == 150:
#         motor2.setPosition(-0.1) # blue 축
#     elif counter == 250:
#         motor3.setPosition(-0.1) # green 축
#         
# 
# # from controller import Robot
# # from math import cos, sin
# 
# # robot = Robot()
# # time_step = int(robot.getBasicTimeStep())
# 
# 
# # # 모터 디바이스 획득
# # motor1 = robot.getDevice('motor 1')
# # motor2 = robot.getDevice('motor 2')
# # motor3 = robot.getDevice('motor 3')
# 
# # # get a ball info
# # ball = robot.getDevice('Ball')
# 
# # # 위치 제어 모드 활성화
# # motor1.setPosition(0.0)
# # motor2.setPosition(0.0)
# # motor3.setPosition(0.0)
# 
# # counter = 0
# # data_dict = {}
# # while robot.step(time_step) != -1:
# #     data_dict['ball_x'] = ball.getField("translation").getSFVec3f()[0]
# #     data_dict['ball_y'] = ball.getField("translation").getSFVec3f()[1]
# #     data_dict['ball_z'] = ball.getField("translation").getSFVec3f()[2]
# #     print(f"ball x: {data_dict['ball_x']}, ball y = {data_dict['ball_y']}, ball z = {data_dict['ball_z']}")
# #     counter += 1
# #     if counter == 50:
# #         motor1.setPosition(-1.0) # red 축
# #     elif counter == 150:
# #         motor2.setPosition(-1.0) # blue 축
# #     elif counter == 250:
# #         motor3.setPosition(1.0) # green 축
