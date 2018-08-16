from PIL import Image, ImageDraw, ImageFont
import random as rd



def gen_letter(letter, size=(32, 32), info=False):
    W, H = size

    # new 32x32 Image with 0 or 1 as pixal value (black & white)
    # 4 times bigger so that quality doesn't get lost while creating
    img = Image.new('1', (W*4, H*4), 0)
    arial = ImageFont.truetype('Arial.ttf', rd.randrange(10*4, 30*4))
    draw = ImageDraw.Draw(img)

    # use textsize function to write in center
    w, h = draw.textsize(letter, font=arial)
    # write letter
    draw.text(((W*4 - w) / 2, (H*4 - h) / 2), letter, fill=1, font=arial)

    # rotate letter
    angle = rd.randrange(360)
    img = img.rotate(angle).resize((W, H))

    print('letter:', letter, 'size:', arial.size//4, 'angle:', angle) if info else None
    return (img, letter)

for char in [chr(i) for i in range(ord("A"), ord("Z")+1)] + [" "]:
    #gen_letter(info=True).resize((256, 256)).show()
    print(char)
    for i in range(2000, 2200):
            
        img, let = gen_letter(char)
        if let == " ":
            let = "SPACE"
        img.save('new_test/' + let + str(i) + '.png')
        print(i)