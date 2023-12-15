import csv
from collections import defaultdict
from os import listdir
from os.path import isfile, join, expanduser
import pandas as pd
from datetime import datetime

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics


masses = defaultdict(list)
angles = (5, 13, 16, 21)

pulse = 7.0
speed = 0.71
speed_first = 0 #int(input("Speed first or not? 1 for yes, and 0 for no: "))
freq = 0 #int(input("Which frequency? "))
no = 10
an_angle = 21 #int(input("Which angle? "))

#num_of_experiments = 4
format = '%Y-%m-%d %H:%M:%S.%f'

valid = f"D:/fluxana/FX_Niryo2/exp/valid2g/{pulse}s/"
nvalid = f"D:/fluxana/FX_Niryo2/exp/not_valid2g/{pulse}s/"
bvalid = "D:/fluxana/FX_Niryo2/exp/22-11/" #+ "valid2g/"

path = expanduser(bvalid)
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

headers = ['iters','timestamp','freq[Hz]', 'angle[deg]', 'pulse_duration [s]', 'dosed[g]', 'mass_flow[g/s]', 'm_left[g]']
#df = pd.read_csv(path + onlyfiles[0])#,names=headers)

def get_summary():
    """This function is to get the average of the mass flow of the same angle with different iterations.
    example of the summery variable for the 4 angles with 8 trails each:
    {5: {1: {'duration': 44.8, 'num_of_iters': 7, 'mass_err': 2.5}, ...., 8: {}, 
    'duration': {'min':---, 'max': ---, 'avg': ---}, 'num_of_iters': {'min': --, 'max': --, 'avg': --}, 'mass_err': {'min': --, 'max': --, 'avg': --}}, 13: {same}, 16: {same}, 21: {same}}"""

    for i in onlyfiles:
        df = pd.read_csv(path + i)
        masses[i] = df

    summury = {5: {}, 13: {}, 16: {}, 21: {}}
    for j in angles:
        for i in range(1,9,1):
            all8 = masses[f"1Hz_1.0speed_100%_{j}deg_2.0grames_pulse-1seconds_{i}.csv"]
            duration = datetime.strptime(all8["timestamp"].iloc[-1], format) - datetime.strptime(all8["timestamp"].iloc[0], format)
            mass_err = all8["dosed[g]"].iloc[-1] - 1.0
            num_of_iters = all8["iters"].iloc[-1]
            summury[j][i]= {"duration": round(duration.total_seconds(), 5), "num_of_iters": num_of_iters, "mass_err": round(mass_err*1000, 3)}

        # Duration, Iterration, and Error's min, max, and avg.   
        min_duration = round(min([summury[j][i]['duration'] for i in summury[j]]), 5)
        max_duration = round(max([summury[j][i]['duration'] for i in summury[j]]), 5)
        avg_duration = round(sum([summury[j][i]['duration'] for i in summury[j]])/len(summury[j]), 5)

        min_iters = round(min([summury[j][i]['num_of_iters'] for i in summury[j]]), 5)
        max_iters = round(max([summury[j][i]['num_of_iters'] for i in summury[j]]), 5)
        avg_iters = round(sum([summury[j][i]['num_of_iters'] for i in summury[j]])/len(summury[j]), 5)

        min_err = round(min([summury[j][i]['mass_err'] for i in summury[j]]), 5)
        max_err = round(max([summury[j][i]['mass_err'] for i in summury[j]]), 5)
        avg_err = round(sum([summury[j][i]['mass_err'] for i in summury[j]])/len(summury[j]), 5)

        summury[j]["duration"] = {"min": min_duration, "max": max_duration, "avg": avg_duration}
        summury[j]["num_of_iters"] = {"min": min_iters, "max": max_iters, "avg": avg_iters}
        summury[j]["mass_err"] = {"min": min_err, "max": max_err, "avg": avg_err}

        print(f"Anlge {j} Summery:")
        print(f'Duration: min = {summury[j]["duration"]["min"]}s, max = {summury[j]["duration"]["max"]}s, Avg = {summury[j]["duration"]["avg"]}s = {round(summury[j]["duration"]["avg"]/60, 5)}min')
        print(f'Iterration: min = {summury[j]["num_of_iters"]["min"]}, max = {summury[j]["num_of_iters"]["max"]}, Avg = {summury[j]["num_of_iters"]["avg"]}')
        print(f'Error: min = {summury[j]["mass_err"]["min"]}mg, max = {summury[j]["mass_err"]["max"]}mg, Avg = {summury[j]["mass_err"]["avg"]}mg\n')

    return masses, summury

masses, summury = get_summary()
#print(list(masses[f"1Hz_1.0speed_100%_{5}deg_2.0grames_pulse-1seconds_{3}.csv"]["dosed[g]"]))
#exit()
def sign(num):
    if num >= 0:
        sign = "+"
    else:
        sign = "-"
    return sign

def average_angle(angle=an_angle):
    global avg_deg, approx, avg_dosed, avg_flow, x_predict, y_predict, fig
    fig=plt.figure(figsize=(16,9), dpi=150,facecolor=(0.8,0.8,0.8))
    gs=gridspec.GridSpec(1,1)
    exp = 8	# The experiment number to be used for the polynomial regression model.
    
    if angle == 5:
        rot_num  = 4
    elif angle == 13 or angle == 16:
        rot_num = 3
    elif angle == 21:
        rot_num = 2

    last_rotation = masses[f"1Hz_1.0speed_100%_{angle}deg_2.0grames_pulse-1seconds_{exp}.csv"]["dosed[g]"].iloc[rot_num]

    dosed = []
    flow = []
    for i in range(1,9,1):
        all8 = masses[f"1Hz_1.0speed_100%_{angle}deg_2.0grames_pulse-1seconds_{i}.csv"]
        dosed.append(list(all8["dosed[g]"]))
        flow.append(list(all8["mass_flow[g/s]"]))
        
    avg_dosed = np.array(masses[f"1Hz_1.0speed_100%_{angle}deg_2.0grames_pulse-1seconds_{exp}.csv"]["dosed[g]"])
    avg_flow = np.array(masses[f"1Hz_1.0speed_100%_{angle}deg_2.0grames_pulse-1seconds_{exp}.csv"]["mass_flow[g/s]"])
    avg_err = summury[angle]["mass_err"]["avg"]

    ax0=fig.add_subplot(gs[:,:], facecolor=(0.9,0.9,0.9))
    avg_deg, = ax0.plot(avg_dosed, avg_flow, "+", label="Actual Data Points", linewidth=1.5)

    ax0.plot(np.array([last_rotation]*len(avg_flow)), avg_flow, "--", label="Last Rotation")
    #plt.axvline(x=0.22058956, color="red", label="Last Rotation")
    ax0.plot(np.array([1]*len(avg_flow)), avg_flow, "k", label="At 1 gram")
    #err5 = ax0.errorbar(avg_dosed, avg_flow, label="Error bar of each point", fmt='none')
    plt.xlim(0, 1.3)
    plt.ylim(0, max(avg_flow) + max(avg_flow)/20)
    plt.ylabel('Mass flow [g/s]', fontsize=12)
    plt.xlabel('Dosed Mass [g]', fontsize=12)

    # Polynomial Regression Algorithm
    #x_train = avg_dosed.reshape(-1, 1)
    #y_train = avg_flow.reshape(-1, 1)

    x_train = np.array(dosed[0] + dosed[1] + dosed[2] + dosed[3] + dosed[4] + dosed[5] + dosed[6] + dosed[7]).reshape(-1, 1)
    y_train = np.array(flow[0] + flow[1] + flow[2] + flow[3] + flow[4] + flow[5] + flow[6] + flow[7]).reshape(-1, 1)

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(x_train)

    reg = LinearRegression()
    reg.fit(X_poly, y_train)

    x_predict = np.arange(0,1.1, 0.1).reshape(-1, 1)

    x_vals = poly_features.transform(x_predict)
    y_predict = reg.predict(x_vals)

    approx, = ax0.plot(x_predict, y_predict, color="g", label="Polynomial Regression Model")
    plt.legend(loc="upper right")

    # Model evaluation using R-Square for Polynomial Regression
    #r_square = metrics.r2_score(y_train1, y_predict[:len(y_train1)])
    coef = list(reg.coef_[0]) + [float(reg.intercept_)]
    
    label = f"Y ="
    for c in coef:
        label += f" {sign(c)} {round(abs(c), 4)}x^{len(coef)-1 - coef.index(c)}"

    #reg_label = f"Y = {round(abs(coef[0]), 4)}*X^4 {sign(coef[1])} {round(abs(coef[1]), 4)}*X^3 {sign(coef[2])} {round(abs(coef[2]), 4)}*X^2 {sign(coef[3])} {round(abs(coef[3]), 4)}*X {sign(coef[4])} {round(abs(coef[4]), 4)}"
    plt.title(f"The mass flow of angle {angle}deg, with {rot_num} iterations of rotation for extensive dosing at {1}rad/s, with average error of {avg_err}mg, and around {round(summury[angle]['duration']['avg'], 2)}s avg. duration of the experiment \n \
              Polynomial Regression Model of the graph below: {label}", fontsize=9)
    print(label)
    #plt.show()

average_angle(21)

def update_plot(num):
    avg_deg.set_data(avg_dosed[0:num], avg_flow[0:num])
    approx.set_data(x_predict[0:num], y_predict[0:num])

    return avg_deg, approx

#mass_ani = animation.FuncAnimation(fig, update_plot,
                                   #frames=100, interval=60, repeat=False, blit=True)
#plt.savefig(name + '.png')
plt.show()

"""
if an_angle:

    name = f"{freq}Hz_{speed}speed_0%_{an_angle}deg_2.0grames_pulse-{pulse}seconds_{no}.csv"


def get_mass(path, speed_first = True, freq=1):
    global name
    
    for j in range(4):

        if an_angle:

            name = f"{freq}Hz_{speed}speed_0%_{an_angle}deg_2.0grames_pulse-{pulse}seconds_{no+j}.csv"
        
        else:

            if speed_first:
                if freq == 0:
                    name = f"{speed}speed_{freq}Hz_0%_{angles[j]}deg_2.0grames_pulse-{pulse}seconds_{no}.csv"#_{j}.csv"
                else:
                    name = f"{speed}speed_{freq}Hz_100%_{angles[j]}deg_2.0grames_pulse-{pulse}seconds_{no}.csv"#_{j}.csv"
            else:
                if freq == 0:
                    name = f"{freq}Hz_{speed}speed_0%_{angles[j]}deg_2.0grames_pulse-{pulse}seconds_{no}.csv"#_{j}.csv"
                else:
                    name = f"{freq}Hz_{speed}speed_100%_{angles[j]}deg_2.0grames_pulse-{pulse}seconds_{no}.csv"#_{j}.csv"

        with open(path + name, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for i in reader:
                masses[angles[j]].append((round(float(i[-3]),5), round(float(i[-2]),5), round(float(i[-1]),5)))

    return masses[5], masses[13], masses[16], masses[21]

angle5, angle13, angle16, angle21 = get_mass(path, speed_first, freq)

dosed5 = np.array([i[0] for i in angle5])
flow5 = np.array([i[1] for i in angle5])
left5 = np.array([i[2] for i in angle5])
iters5 = np.arange(1, len(flow5)+1, 1)

dosed13 = np.array([i[0] for i in angle13])
flow13 = np.array([i[1] for i in angle13])
left13 = np.array([i[2] for i in angle13])
iters13 = np.arange(1, len(flow13)+1, 1)

dosed16 = np.array([i[0] for i in angle16])
flow16 = np.array([i[1] for i in angle16])
left16 = np.array([i[2] for i in angle16])
iters16 = np.arange(1, len(flow16)+1, 1)

dosed21 = np.array([i[0] for i in angle21])
flow21 = np.array([i[1] for i in angle21])
left21 = np.array([i[2] for i in angle21])
iters21 = np.arange(1, len(flow21)+1, 1)

max_flow = max(list(flow5) + list(flow13) + list(flow16) + list(flow21))

#min_dosed = min(list(dosed5) + list(dosed13) + list(dosed16) + list(dosed21))
dosed = [list(dosed5), list(dosed13), list(dosed16), list(dosed21)]
flow = [list(flow5), list(flow13), list(flow16), list(flow21)]
length = [len(dosed5), len(dosed13), len(dosed16), len(dosed21)]

# Getting the maximum error in the dosed axis (x-axis).
xerror = []
xerr = 0
for i in range(min(length)):
    for j in range(3):
        if abs(dosed[j][i] - dosed[j+1][i]) > xerr:
            xerr = abs(dosed[j][i] - dosed[j+1][i])
    xerror.append(xerr)
    xerr = 0
print(xerror)

def edit_csv():
    with open(path + name, "w") as f:
        wr = csv.reader(f)
        for i in wr:
            for j in i:
                i[i.index(j)] = j[:j.find("[")]
            #writer.writerow(i)
            print(i)
            break

data = pd.read_csv(path + name)
#print(data.loc[0,"dosed[g]"])

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    
    n = len(points[0])

    for i in range(n):
        #x = points.loc(i, "dosed[g]")
        #y = points.loc(i, "mass_flow[g/s]")
        
        x = float(points[0][i])
        y = float(points[1][i])

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

def sign(num):
    if num >= 0:
        sign = "+"
    else:
        sign = "-"
    return sign

def linear_reg(points):
    m_now = 0.04
    b_now = 0
    L = 0.00001
    epochs = 10000

    for i in range(epochs):
        m, b = gradient_descent(m_now, b_now, points, L)
    return m, b

def average_angle(angle=an_angle):
    fig=plt.figure(figsize=(16,9), dpi=150,facecolor=(0.8,0.8,0.8))
    gs=gridspec.GridSpec(1,1)

    avg_dosed = []
    avg_flow = []

    for i in range(min(length)):
        avg_dosed.append((dosed[0][i] + dosed[1][i] + dosed[2][i] + dosed[3][i])/4)
        avg_flow.append((flow[0][i] + flow[1][i] + flow[2][i] + flow[3][i])/4)
        
    avg_dosed = np.array(avg_dosed)
    avg_flow = np.array(avg_flow)

    ax0=fig.add_subplot(gs[:,:], facecolor=(0.9,0.9,0.9))
    avg_deg = ax0.scatter(avg_dosed, avg_flow, label="Actual Data Points")#, 'k+-', linewidth=1.5)
    ax0.plot(np.array([1]*len(avg_flow)), avg_flow, label="At 1 gram")
    err5 = ax0.errorbar(avg_dosed, avg_flow, xerr=xerror, ecolor="red", label="Error bar of each point", fmt='none')
    plt.xlim(0, 2)
    plt.ylim(0, max(avg_flow) + max(avg_flow)/20)
    plt.ylabel('Mass flow [g/s]', fontsize=12)
    plt.xlabel('Dosed Mass [g]', fontsize=12)

    # Polynomial Regression Algorithm
    x_train = avg_dosed.reshape(-1, 1)
    y_train = avg_flow.reshape(-1, 1)

    x_train1 = np.array(dosed[0] + dosed[1] + dosed[2] + dosed[3]).reshape(-1, 1)
    y_train1 = np.array(flow[0] + flow[1] + flow[2] + flow[3]).reshape(-1, 1)

    poly_features = PolynomialFeatures(degree=4, include_bias=False)
    X_poly = poly_features.fit_transform(x_train1)

    reg = LinearRegression()
    reg.fit(X_poly, y_train1)

    x_predict = np.arange(0,2.1, 0.1).reshape(-1, 1)

    x_vals = poly_features.transform(x_predict)
    y_predict = reg.predict(x_vals)

    plt.plot(x_predict, y_predict, color="g", label="Polynomial Regression Model")
    plt.legend(loc="upper right")

    # Model evaluation using R-Square for Polynomial Regression
    #r_square = metrics.r2_score(y_train1, y_predict[:len(y_train1)])
    coef = list(reg.coef_[0]) + [float(reg.intercept_)]
    
    label = f"Y ="
    for c in coef:
        label += f" {sign(c)}{round(abs(c), 4)}X^{len(coef)-1 - coef.index(c)}"

    #reg_label = f"Y = {round(abs(coef[0]), 4)}*X^4 {sign(coef[1])} {round(abs(coef[1]), 4)}*X^3 {sign(coef[2])} {round(abs(coef[2]), 4)}*X^2 {sign(coef[3])} {round(abs(coef[3]), 4)}*X {sign(coef[4])} {round(abs(coef[4]), 4)}"
    plt.title(f"The average mass flow of values with {angle}deg, {freq}Hz, {speed}rad/s, No. {min(length)} iterations, and max. dosed error is {round(max(xerror), 3)}g \n \
              Polynomial Regression Model: {label}", fontsize=9)
    print(label)

def update_plot(num):
    deg5.set_data(dosed5[0:num], flow5[0:num])
    deg13.set_data(dosed13[0:num], flow13[0:num])
    deg16.set_data(dosed16[0:num], flow16[0:num])
    deg21.set_data(dosed21[0:num], flow21[0:num])

    return deg5, deg13, deg16, deg21


x_axis = 20
 
def four_grids():
    global deg5, deg13, deg16, deg21, fig

    fig=plt.figure(figsize=(16,9), dpi=150,facecolor=(0.8,0.8,0.8))
    gs=gridspec.GridSpec(2,2)
    
    ax0=fig.add_subplot(gs[0,0], facecolor=(0.9,0.9,0.9))
    #deg5, = ax0.plot(iters5, flow5, 'k+-', linewidth=1.5)
    #plt.xlim(1, x_axis)]
    deg5, = ax0.plot(dosed5, flow5, 'k+-', linewidth=1.5)
    ax0.plot(np.array([1]*len(flow5)), flow5)
    #err5 = ax0.errorbar(dosed5, flow5, xerr=xerror)
    plt.xlim(0, 2)
    plt.ylim(0, max_flow)

    #plt.xticks(range(1,101,1))

    plt.title(f"At -5 degrees with no. {len(dosed5)} iterations, & with {left5[-1]}g mass left, and {round(1-left5[-1],4)}g error", fontsize=10)
    ##plt.title(f"No. {len(dosed5)} iterations, & with {left5[-1]}g mass left", fontsize=10)

    #plt.ylabel('Mass flow [g/s]', fontsize=12)
    fig.text(0.05, 0.45, 'Mass flow [g/s]', fontsize=12, rotation='vertical', ha='center')

    ax1=fig.add_subplot(gs[0,1], facecolor=(0.9,0.9,0.9))
    #deg13, = ax1.plot(iters13, flow13, 'g+-', linewidth=1.5)
    #plt.xlim(1, x_axis)
    deg13, = ax1.plot(dosed13, flow13,  'g+-', linewidth=1.5)
    ax1.plot(np.array([1]*len(flow13)), flow13)
    plt.xlim(0, 2)
    plt.ylim(0, max_flow)
    plt.title(f"At -13 degrees with no. {len(dosed13)} iterations, & with {left13[-1]}g mass left, and {round(1-left13[-1],4)}g error", fontsize=10)
    ##plt.title(f"No. {len(dosed13)} iterations, & with {left13[-1]}g mass left", fontsize=10)

    ax2=fig.add_subplot(gs[1,0], facecolor=(0.9,0.9,0.9))
    #deg16, = ax2.plot(iters16, flow16, 'r+-', linewidth=1.5)
    #plt.xlim(1, x_axis)
    deg16, = ax2.plot(dosed16, flow16,  'r+-', linewidth=1.5)
    ax2.plot(np.array([1]*len(flow16)), flow16)
    plt.xlim(0, 2)
    plt.ylim(0, max_flow)
    plt.title(f"At -16 degrees with no. {len(dosed16)} iterations, & with {left16[-1]}g mass left, and {round(1-left16[-1],4)}g error", fontsize=10)
    ##plt.title(f"No. {len(dosed16)} iterations, & with {left16[-1]}g mass left", fontsize=10)

    fig.text(0.52, 0.04, 'Dosed Mass [g]', fontsize=12, ha='center')

    ax3=fig.add_subplot(gs[1,1], facecolor=(0.9,0.9,0.9))
    #deg21, = ax3.plot(iters21, flow21,'b+-', linewidth=1.5)
    #plt.xlim(1, x_axis)
    deg21, = ax3.plot(dosed21, flow21, 'b+-', linewidth=1.5)
    ax3.plot(np.array([1]*len(flow21)), flow21)
    plt.xlim(0, 2)
    plt.ylim(0, max_flow)
    plt.title(f"At -21 degrees with no. {len(dosed21)} iterations, & with {left21[-1]}g mass left, and {round(1-left21[-1],4)}g error", fontsize=10)
    ##plt.title(f"No. {len(dosed21)} iterations, & with {left21[-1]}g mass left", fontsize=10)

#four_grids()

#average_angle()

#mass_ani = animation.FuncAnimation(fig, update_plot,
                                   #frames=100, interval=60, repeat=False, blit=True)
#plt.savefig(name + '.png')
#plt.show()

"""