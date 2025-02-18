# Arduino Fundamentals
**Arduino**
- an open-source electronics platform
- Arduino boards are able to read inputs-light on a sensor, a finger on a button or a Twitter message - and turn it into an output - activating a motor, turnning on an LED

---
# Resources
- [Arduino Documentation](https://docs.arduino.cc/)
- [Arduino Cloud](https://docs.arduino.cc/arduino-cloud/)
- [Official Tutorials](https://docs.arduino.cc/tutorials/)
- [Learn Arduino](https://docs.arduino.cc/learn/)

---
# Key components
![Arduino Board](pic_Arduino/Arduino_Board.png)
- **Microcontroller**
  - the brain of Arduino
  - the component that we load programs into
  - designed to execute only a specific number of things
- **USB port**
  - used to connect Arduino board to a computer
- **USB to Serial chip**
  - helps translate data that comes from e.g. a computer to the on-board microcontroller
  - this is what makes it possible to program the Arduino board from computer
- **Digital pins**
  - use digital logic (0, 1 or LOW/HIGH)
  - commonly used for switches and to turn on/off an LED
- **Analog pins**
  - pins that can read analog values in a 10 bit resolution (0-1023)
- **5V / 3.3V pins**
  - used to power external components
- **GND**
  -  also known as `ground`, `negative` or simply `-`, is used to complete a circuit, where the electrical level is at 0 volt
- **VIN**
  - stands for Voltage In, where you can connect external power supplies

---
# Basic Operation
- The program that is loaded to the microcontroller will start execution as soon as it's powered
- Every program has a function called "**loop**", in which you can:
  - Read a sensor
  - Turn on a light
  - Check whether a condition is met
  - etc.
- The speed of a program is fast, unless tell it to slow down
  - depends on the sice of the program and how long it takes for the microcontroller to execute it

The basic operation of an Arduino:
![Basic Operation of an Arduino](pic_Arduino/Basic_Operation.png)


## Communication through Arduino
- **Inputs** ➡️ **Sensors**
  - Take inputs from the physical and turn it into electronic signals
  - Push putton, light-dependent resistors/phototransistor
- **Outputs** ➡️ **Actuators**
  - Take electric signals and perform an action
  - LEDs, motors, speakers ...

<img src='pic_Arduino/input_output.png' style="width: 100%; max-width: 60%" />

- **Digital Inputs** (数字输入)
  - 上图为**按钮**，仅提供按下（ON）或未按下（OFF）
  - 按下时发送高电平信号（HIGH）
  - 松开时发送低电平信号（LOW）
- **Analog Inputs** (模拟输入)
  - 图中为**电位器**
  - 电位器提供连续的电压值，随旋钮转动而变化
  - Arduino读取这个电压值来确定旋钮的位置，处理模拟信号
- **Gigital Outputs** (数字输出)
  - 图中LED，通常只有两种状态：亮（ON）或灭（OFF）
  - Arduino通过发送高电平或低电平信号来控制LED的亮灭
- **Analog Outputs** (模拟输出)
  - 图中**蜂鸣器**可以产生不同频率的声音
  - Arduino通过发送脉宽调制（Pulse-Width Modulation: PWM）信号来控制蜂鸣器的音调和音量，处理模拟信号

### Electronic Signals
#### Analog Signal
==模拟信号是**连续**的==
- is generally bound to a range
- typically 0-5V, or 0-3.3V in an Arduino
![basics_analog_signal](pic_Arduino/basics_analog_signal.png)

#### Digital Signal
==数字信号是**离散**的==
- represents only two **binary states** (**0 or 1**)
  - are read as **high** or **low** states in the program
  - the most common signal type in modern technology
![basics_digital_signal](pic_Arduino/basics_digital_signal.png)

## Circuit Basics
**LED Circuit**: an example of a circuit
![LED circuit with an Arduino](pic_Arduino/LED_Circuit.png)
- <div style="color: grey;">Arduino控制LED的亮灭。电路中，LED的正极连接到Arduino的一个数字引脚，负极通过一个电阻接到地（GND）。电阻的作用是保护LED，防止电流过大烧坏LED。当引脚设置为高电平（HIGH）时，电流通过电路，LED亮起；当引脚设置为低电平（LOW）时，电流不通过电路，LED熄灭</div>


---
# Programming Language
- based on Wiring, and the Arduino Software (IDE), based on Processing


---