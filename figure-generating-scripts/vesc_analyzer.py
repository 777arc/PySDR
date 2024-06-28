import numpy as np
import matplotlib.pyplot as plt
import csv

log_file = '/mnt/c/Users/marclichtman/Downloads/2024-06-12_00-34-31.csv' # 2024-06-12_00-27-18.csv

ms_today = []
input_voltage = []
temp_mos_max = []
temp_mos_1 = []
temp_mos_2 = []
temp_mos_3 = []
temp_motor = []
current_motor = []
current_in = []
d_axis_current = []
q_axis_current = []
erpm = []
duty_cycle = []
amp_hours_used = []
amp_hours_charged = []
watt_hours_used = []
watt_hours_charged = []
tachometer = []
tachometer_abs = []
encoder_position = []
fault_code = []
vesc_id = []
d_axis_voltage = []
q_axis_voltage = []
ms_today_setup = []
amp_hours_setup = []
amp_hours_charged_setup = []
watt_hours_setup = []
watt_hours_charged_setup = []
battery_level = []
battery_wh_tot = []
current_in_setup = []
current_motor_setup = []
speed_meters_per_sec = []
tacho_meters = []
tacho_abs_meters = []

#x = np.genfromtxt(log_file, delimiter=',', skip_header=1)
with open(log_file, newline='') as csvfile:
    for row in csv.reader(csvfile, delimiter=';', quotechar='|'):
        if row[0] == 'ms_today':
            continue
        ms_today.append(row[0])
        input_voltage.append(row[1])
        temp_mos_max.append(row[2])
        temp_mos_1.append(row[3])
        temp_mos_2.append(row[4])
        temp_mos_3.append(row[5])
        temp_motor.append(row[6])
        current_motor.append(row[7])
        current_in.append(row[8])
        d_axis_current.append(row[9])
        q_axis_current.append(row[10])
        erpm.append(row[11])
        duty_cycle.append(row[12])
        amp_hours_used.append(row[13])
        amp_hours_charged.append(row[14])
        watt_hours_used.append(row[15])
        watt_hours_charged.append(row[16])
        tachometer.append(row[17])
        tachometer_abs.append(row[18])
        encoder_position.append(row[19])
        fault_code.append(row[20])
        vesc_id.append(row[21])
        d_axis_voltage.append(row[22])
        q_axis_voltage.append(row[23])
        ms_today_setup.append(row[24])
        amp_hours_setup.append(row[25])
        amp_hours_charged_setup.append(row[26])
        watt_hours_setup.append(row[27])
        watt_hours_charged_setup.append(row[28])
        battery_level.append(row[29])
        battery_wh_tot.append(row[30])
        current_in_setup.append(row[31])
        current_motor_setup.append(row[32])
        speed_meters_per_sec.append(row[33])
        tacho_meters.append(row[34])
        tacho_abs_meters.append(row[35])


t = np.array(ms_today).astype(np.float32) / 1000.0 # convert to seconds
t -= t[0] # start at 0

current_in = np.array(current_in).astype(np.float32)
current_motor = np.array(current_motor).astype(np.float32)
temp_mos_max = np.array(temp_mos_max).astype(np.float32)
temp_mos_1 = np.array(temp_mos_1).astype(np.float32)
temp_mos_2 = np.array(temp_mos_2).astype(np.float32)
temp_mos_3 = np.array(temp_mos_3).astype(np.float32)
temp_motor = np.array(temp_motor).astype(np.float32)
speed_meters_per_sec = np.array(speed_meters_per_sec).astype(np.float32)
speed_miles_per_hour = speed_meters_per_sec * 2.23694


plt.figure(0)
plt.plot(t, current_motor, 'g')
plt.plot(t, current_in, 'b')
plt.plot(t, temp_mos_max, 'r')
plt.xlabel('Time (s)')
plt.legend(['Input Current [A]', 'Motor Current [A]', 'FET Temp [C]'])
plt.grid()

plt.figure(1)
plt.plot(speed_miles_per_hour)

plt.show()