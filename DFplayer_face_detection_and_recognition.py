import sensor, image, lcd
import KPU as kpu
from Maix import GPIO
from fpioa_manager import *
from machine import UART
import time

fm.register(20, fm.fpioa.UART2_TX) # Declare the pin for TX uart
fm.register(21, fm.fpioa.UART2_RX) # Declare the pin for RX uart

uart = UART(UART.UART2, 9600, 8, None, 1, timeout = 1000, read_buf_len = 4096)

# Commands for DFplayer
cmd_init = bytearray([0x7E, 0xFF, 0x06, 0x0C, 0x01, 0x00, 0x00, 0xFE, 0xEE, 0xEF])
cmd_vol =  bytearray([0x7E, 0xFF, 0x06, 0x06, 0x01, 0x00, 0x14, 0xFE, 0xE0, 0xEF]) # 14 in HEX is 20 in DEC
cmd_1st_song = bytearray([0x7E, 0xFF, 0x06, 0x03, 0x01, 0x00, 0x01, 0xFE, 0xF6, 0xEF])
cmd_2nd_song = bytearray([0x7E, 0xFF, 0x06, 0x03, 0x01, 0x00, 0x02, 0xFE, 0xF5, 0xEF])
cmd_3rd_song = bytearray([0x7E, 0xFF, 0x06, 0x03, 0x01, 0x00, 0x03, 0xFE, 0xF4, 0xEF])
cmd_4th_song = bytearray([0x7E, 0xFF, 0x06, 0x03, 0x01, 0x00, 0x04, 0xFE, 0xF3, 0xEF])
cmd_5th_song = bytearray([0x7E, 0xFF, 0x06, 0x03, 0x01, 0x00, 0x05, 0xFE, 0xF2, 0xEF])

cmd_dict = {'Angry' : cmd_1st_song, 'Happy' : cmd_2nd_song, 'Sad' : cmd_3rd_song, 'Surprise' : cmd_4th_song}

uart.write(cmd_init)
uart.read()
uart.write(cmd_vol)
uart.read()

lcd.init(freq=15000000)
sensor.reset()                      # Reset and initialize the sensor. It will
                                    # run automatically, call sensor.run(0) to stop
sensor.set_pixformat(sensor.RGB565) # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)   # Set frame size to QVGA (320x240)
#sensor.set_windowing((320, 240))
sensor.skip_frames(time = 2000)     # Wait for settings take effect.
sensor.set_vflip(1)
sensor.set_hmirror(1)
clock = time.clock()                # Create a clock object to track the FPS.

print('loading face detect model')
task_detect_face = kpu.load(0x300000) # Charge face detect model into KPU
print('loading face expresion classify model')
task_classify_face = kpu.load(0x500000) # Charge face classification model into KPU

a=kpu.set_outputs(task_classify_face, 0, 1, 1, 5)

anchor = (1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437, 6.92275, 6.718375, 9.01025)
a = kpu.init_yolo2(task_detect_face, 0.5, 0.3, 5, anchor)

labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise'] # Facial expression labels

print('configuration complete')

while(True):
    clock.tick()                    # Update the FPS clock.
    img = sensor.snapshot()         # Take a picture and return the image.
    detected_face = kpu.run_yolo2(task_detect_face, img)

    if detected_face:
        for i in detected_face:
            face = img.cut(i.x(), i.y(), i.w(), i.h())
            face_128 = face.resize(128, 128)
            a = face_128.pix_to_ai()
            fmap = kpu.forward(task_classify_face, face_128)
            plist = fmap[:]
            pmax = max(plist)
            #print("%s: %s" % (labels[plist.index(pmax)], pmax))
            print(plist)
            label = labels[plist.index(pmax)]

            if label != 'Neutral':
                uart.write(cmd_dict[label])
                uart.read()
                time.sleep(2)
