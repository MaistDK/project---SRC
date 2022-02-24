import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import foier

gyro = pd.read_csv("Gyroscope.csv")
accel = pd.read_csv("Linear Accelerometer.csv")

velx = [0]
vely = [0]
posx = [0]
posy = [0]
time = [0]

dir = np.array([[0,1]])
print(dir)

start_time = 50
end_time = start_time + 34.8+5

mass = 123.8 * 1000

def energy(t, px, py, vx, vy, ax, ay):
    cp = 1.05
    A = 3.52 * 4.3
    abs_pos = []
    abs_vel = []
    acc_force = []
    for x, y in zip(px, py):
        abs_pos.append(np.sqrt(x**2+y**2))
    for x, y in zip(vx, vy):
        abs_vel.append(np.sqrt(x**2+y**2))
    for x, y in zip(ax, ay):
        acc_force.append(np.sqrt(x**2+y**2) * mass)

    acc_energy = num_int(abs_pos, acc_force)

    drag_force = []
    for vel in abs_vel:
        drag_force.append(0.5 * cp * A * vel ** 2)

    drag_energy = num_int(abs_pos, drag_force)

    total_energy = drag_energy[-1] + acc_energy[-1]

    print(f"total_energy: {total_energy}")
    print(f"drag_energy: {drag_energy}")

    return(acc_energy, drag_energy)


def map():
    global dir, velx, vely, posx, posy, time

    rotz = [0]

    tvec = [[0,0]]
    accx = []
    accy = []

    pos = [[0, 0]]
    corrected = [[0, 0]]
    ac = [[0, 0]]
    #pos = [[0,0]]

    #i = 0
    #vel = np.array([startV, 0])


    for index, rows in accel.iterrows():
        index = int(index)
        t = rows['Time (s)']

        if t >= start_time and t<end_time:
            time.append(t-start_time)
            ac.append([rows['X (m/s^2)'],rows['Y (m/s^2)']])

    #print(time)

    #print(time)

    for index, row in gyro.iterrows():
        if row['Time (s)'] > start_time and row['Time (s)'] < end_time:
            rotz.append(row['Z (rad/s)'])

        #if row['Time (s)'] > startTime:'''
    rot = np.pi / 2

    print(len(rotz))
    print(len(time))

    for i in range(len(time)):
        rot = (time[i] - time[i-1]) * ((rotz[i] + rotz[i - 1]) / 2) + rot
        dir = np.concatenate((dir, np.array([[np.cos(rot), np.sin(rot)]])))
        tvec.append([-dir[-1][1], dir[-1][0]])

        direction = np.array([np.cos(rot), np.sin(rot)])
        tvecs = np.array([-dir[-1][1], dir[-1][0]])
        avec1 = direction * ac[i][1]
        avec2 = tvecs * -(ac[i][0])

        vec = avec1 + avec2
        # print(vec)

        accx.append(vec[0])
        accy.append(vec[1])

        '''for j in range(len(rotz)):

            #time.append(time[-1] + 0.002)
            if rotz[j][0] < time[i] and rotz[j][0] > time[i-1]:
                rot = (rotz[j][0] - rotz[j-1][0]) * ((rotz[j][1]+rotz[j-1][1])/2) + rot
                print(f"rotation: {rotz[j]}")

            if rotz[j][0] > time[i]:
                dir = np.concatenate((dir, np.array([[np.cos(rot), np.sin(rot)]])))
                tvec.append([-dir[-1][1], dir[-1][0]])

                direction = np.array([np.cos(rot), np.sin(rot)])
                tvecs = np.array([-dir[-1][1], dir[-1][0]])
                avec1 = direction * ac[i][1]
                avec2 = tvecs * -(ac[i][0])

                vec = avec1 + avec2
                #print(vec)

                accx.append(vec[0])
                accy.append(vec[1])
                break;
        print(f"time: {time[i]}")
            #print(rot)'''

    print(rot)
    print(ac)
    #print(dir[0:100])

    velx = velx + num_int(time, accx)
    vely = vely + num_int(time, accy)
    posx = posx + num_int(time, velx)
    posy = posy + num_int(time, vely)

    #print(corrected)
    data = {'time': pd.Series(time), 'accx': pd.Series(accx), 'accy': pd.Series(accy), 'velx': pd.Series(velx),
            'vely': pd.Series(vely), 'posx':pd.Series(posx), 'posy': pd.Series(posy)}
    df = pd.DataFrame(data)
    print(df.head())
    dire = [i[0] for i in dir[:-1]]

    df.to_excel('file.xlsx', index=False, sheet_name='sheet1', engine='xlsxwriter')


def num_int(t, y):
    area = [0]
    for i in range(len(t)):

        if i > len(y):
            break

        if i > 0:
            area.append((t[i] - t[i - 1]) * ((y[i] + y[i - 1]) / 2) + area[i - 1])
    return(area)


def correction():
    time_dif = time[-1]

    velocity = 26

    scale = velocity / (np.sqrt(velx[-1]**2 + vely[-1]**2))

    print(scale)

    corv_x = []
    corv_y = []

    velx_dif = velx[-1] - (velx[-1] * scale)
    vely_dif = vely[-1] - (vely[-1] * scale)

    x_corr = velx_dif / time_dif
    y_corr = vely_dif / time_dif

    for t,x,y in zip(time, velx, vely):
        corv_x.append(x-(t*x_corr))
        corv_y.append(y-(t*y_corr))

    corp_x = num_int(time, corv_x)
    corp_y = num_int(time, corv_y)

    cora_x = np.gradient(corv_x, time)
    cora_y = np.gradient(corv_y, time)

    #a, d = energy(time, corp_x, corp_y, corv_x, corv_y, cora_x, cora_y)
    absolute_corv = []
    absolute_corp = []

    for x, y in zip(corv_x, corv_y):
        absolute_corv.append(np.sqrt(x**2+y**2))
    for x, y in zip(corp_x, corp_y):
        absolute_corp.append(np.sqrt(x ** 2 + y ** 2))

    absolute_vel = []
    absolute_pos = []

    for x, y in zip(velx, vely):
        absolute_vel.append(np.sqrt(x**2+y**2))
    for x, y in zip(posx, posy):
        absolute_pos.append(np.sqrt(x ** 2 + y ** 2))
    print(absolute_corp[-1])
    print(absolute_pos[-1])
    #foier.series(time, corp_x, corp_y)
    plt.plot(time, absolute_pos[:-1])
    plt.plot(time, absolute_corp)
    plt.ylabel('Absolut position (m)')
    plt.xlabel('Tid (s)')
    #plt.xlim(-500, 0)
    #plt.ylim(0, 500)
    plt.grid()
    plt.show()



map()
correction()


#print(df.head())

