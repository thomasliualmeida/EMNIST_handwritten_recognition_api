import requests
import cv2 as cv

print('CONSUMER API Handwritten Digits')

print('Exemplo 1 - letra S utilizando o endpoint')
url = 'http://127.0.0.1:8015/predict/'
img = cv.imread('data/test/test_S.png', cv.IMREAD_GRAYSCALE)
_, img_encoded = cv.imencode('.png', img)
response = requests.post(url, data=img_encoded.tostring())
print(f'Prediction: : {response.text}')

print('Exemplo 2 - letra a utilizando o endpoint')
url = 'http://127.0.0.1:8015/predict/'
img = cv.imread('data/test/test_a_lower.png', cv.IMREAD_GRAYSCALE)
_, img_encoded = cv.imencode('.png', img)
response = requests.post(url, data=img_encoded.tostring())
print(f'Prediction: : {response.text}')

print('Exemplo 3 - letra B utilizando o endpoint')
url = 'http://127.0.0.1:8015/predict/'
img = cv.imread('data/test/test_B.png', cv.IMREAD_GRAYSCALE)
_, img_encoded = cv.imencode('.png', img)
response = requests.post(url, data=img_encoded.tostring())
print(f'Prediction: : {response.text}')

print('Exemplo 4 - letra C utilizando o endpoint')
url = 'http://127.0.0.1:8015/predict/'
img = cv.imread('data/test/test_C.png', cv.IMREAD_GRAYSCALE)
_, img_encoded = cv.imencode('.png', img)
response = requests.post(url, data=img_encoded.tostring())
print(f'Prediction: : {response.text}')

print('Exemplo 5 - letra d utilizando o endpoint')
url = 'http://127.0.0.1:8015/predict/'
img = cv.imread('data/test/test_d_lower.png', cv.IMREAD_GRAYSCALE)
_, img_encoded = cv.imencode('.png', img)
response = requests.post(url, data=img_encoded.tostring())
print(f'Prediction: : {response.text}')

print('Exemplo 6 - letra E utilizando o endpoint')
url = 'http://127.0.0.1:8015/predict/'
img = cv.imread('data/test/test_E.png', cv.IMREAD_GRAYSCALE)
_, img_encoded = cv.imencode('.png', img)
response = requests.post(url, data=img_encoded.tostring())
print(f'Prediction: : {response.text}')

print('Exemplo 7 - letra F utilizando o endpoint')
url = 'http://127.0.0.1:8015/predict/'
img = cv.imread('data/test/test_F.png', cv.IMREAD_GRAYSCALE)
_, img_encoded = cv.imencode('.png', img)
response = requests.post(url, data=img_encoded.tostring())
print(f'Prediction: : {response.text}')

print('Exemplo 8 - letra A utilizando o endpoint')
url = 'http://127.0.0.1:8015/predict/'
img = cv.imread('data/test/test_A.png', cv.IMREAD_GRAYSCALE)
_, img_encoded = cv.imencode('.png', img)
response = requests.post(url, data=img_encoded.tostring())
print(f'Prediction: : {response.text}')