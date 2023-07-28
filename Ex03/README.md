# Code Peer Review Templete
- 코더 : 백기웅 님
- 리뷰어 : 김재림


# PRT(PeerReviewTemplate)

각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.

- [x] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [x] 주석을 보고 작성자의 코드가 이해되었나요?
- [ ] 코드가 에러를 유발할 가능성이 있나요?
- [x] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
- [x] 코드가 간결한가요?

# 참고 링크 및 코드 개선 여부

1. 모델 완성이 잘 이루어졌습니다.
2. 코드에 대해 이해를 잘 하시고, 설명도 잘 해주셨습니다.
3. 아래의 코드에서 세장의 사진에서 한장의 사진만 출력이 되었는데, 이 부분만 개선되면 좋겠습니다.

```python
# 원본이미지를 img_show에 할당한뒤 이미지 사람이 있는 위치와 배경을 분리해서 표현한 color_mask 를 만든뒤 두 이미지를 합쳐서 출력
img_show = img_orig.copy()

# True과 False인 값을 각각 255과 0으로 바꿔줍니다
img_mask = seg_map.astype(np.uint8) * 255

# 255와 0을 적당한 색상으로 바꿔봅니다
color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)

# 원본 이미지와 마스트를 적당히 합쳐봅니다
# 0.6과 0.4는 두 이미지를 섞는 비율입니다.
img_show = cv2.addWeighted(img_show, 0.6, color_mask, 0.4, 0.0)

plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()
```

