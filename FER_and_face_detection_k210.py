import sensor, image, time, lcd
import KPU as kpu

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

a=kpu.set_outputs(task_classify_face, 0, 1, 1, 2)

anchor = (1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437, 6.92275, 6.718375, 9.01025)
a = kpu.init_yolo2(task_detect_face, 0.5, 0.3, 5, anchor)

labels = ['happy', 'sad'] # Facial expression labels

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
            print("%s: %s" % (labels[plist.index(pmax)], pmax))
