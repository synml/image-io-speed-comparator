# image-io-speed-comparator
> 라이브러리 간의 이미지 로드 속도(성능)을 비교합니다.

## 1. 이미지 읽기 속도 비교
쉘에서 아래의 명령을 실행합니다.

repeat은 반복 실험 횟수를 지정하며, 기본값은 5입니다.
```bash
python read.py --repeat <num_repeat>
```

## 2. 이미지 증강(augmentation) 속도 비교
쉘에서 아래의 명령을 실행합니다.

repeat은 반복 실험 횟수를 지정하며, 기본값은 5입니다.
```bash
python augmentation.py --repeat <num_repeat> --num_workers <num>
```

## (선택적) png 형식 이미지를 jpg로 변환
data 폴더에는 실험에 사용할 이미지가 들어있으며, 이미지를 추가, 수정, 삭제하실 수 있습니다.<br>
그러나 이미지 형식이 jpg가 아니면 jpeg4py에서 로드가 불가능합니다.<br>
따라서 png 형식의 이미지를 jpg로 변환해서 저장해주는 코드를 마련하였습니다.<br>
사용 방법은 data 폴더에 png 이미지들을 넣은 뒤, 쉘에서 아래의 명령을 실행합니다.
```bash
python convert_png_to_jpg.py
```
