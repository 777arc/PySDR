import uhd
import time

# from https://events.gnuradio.org/event/18/contributions/234/attachments/74/186/GPIOs%20on%20USRPs.pdf

usrp = uhd.usrp.MultiUSRP(args="addr=192.168.1.201")
usrp.set_gpio_attr('FP0A', 'CTRL', 0x000, 0xFFF)
usrp.set_gpio_attr('FP0A', 'DDR', 0xFFF, 0xFFF)
for i in range(10):
    print("Off")
    usrp.set_gpio_attr('FP0A', 'OUT', 0x000, 0xFFF)
    time.sleep(1)
    print("On")
    usrp.set_gpio_attr('FP0A', 'OUT', 0xFFF, 0xFFF)
    time.sleep(1)