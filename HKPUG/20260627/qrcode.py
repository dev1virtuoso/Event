import qrcode

url_data = 'https://github.com/dev1virtuoso/Documentation/blob/main/dev1virtuoso/Attachment/dev1virtuoso/carson-wu.md#Contact'

qr = qrcode.QRCode(
    version=None,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)

qr.add_data(url_data)
qr.make(fit=True)

img = qr.make_image(fill_color="purple", back_color="white")

img.save('qrcode.png')

print('QR code generated successfully')
