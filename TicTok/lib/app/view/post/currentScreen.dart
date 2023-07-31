import 'package:flutter/material.dart';
import 'package:tiktok_practice/app/data/service/video_controll.dart';
import 'package:tiktok_practice/app/view/mypage/mypage.dart';
import 'package:tiktok_practice/app/view/post/videoFeed.dart';
import 'package:tiktok_practice/app/view/uploard/upload.dart';

Widget currentScreen() {
  ViedeoControll viedeoControl = ViedeoControll();
  switch (viedeoControl.currentScreen) {
    case 0:
      return feedVideos();
    case 1:
      return ProfileScreen();
    case 2:
      return UploadScreen();
    default:
      return feedVideos();
  }
}
