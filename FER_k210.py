import sensor, image, time, lcd
import KPU as kpu

lcd.init(freq=15000000)
sensor.reset()                      # Reset and initialize the sensor. It will
                                    # run automatically, call sensor.run(0) to stop
sensor.set_pixformat(sensor.RGB565) # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)   # Set frame size to QVGA (320x240)
sensor.set_windowing((224, 224))
sensor.skip_frames(time = 2000)     # Wait for settings take effect.
sensor.set_vflip(1)
sensor.set_hmirror(1)
clock = time.clock()                # Create a clock object to track the FPS.
task = kpu.load(0x300000) # mnist
a=kpu.set_outputs(task, 0, 1, 1, 2)

while(True):
    clock.tick()                    # Update the FPS clock.
    img = sensor.snapshot()         # Take a picture and return the image.
    #img.invert() # Use with numbers in paper
    img_3 = img.resize(128, 128)

    a = img_3.pix_to_ai()
    fmap = kpu.forward(task, img_3)
    plist = fmap[:]
    pmax = max(plist)
    max_index = plist.index(pmax)
    #img.draw_string(5, 5, ('%s' % (max_index)), color = (255, 255, 255), scale = 2)
    lcd.display(img)                # Display on LCD
    #print("%d: %.3f" % (max_index, pmax))
    print(plist)

