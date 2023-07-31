import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';
import 'package:logger/logger.dart';
import 'package:tiktok_practice/app/data/model/video.dart';
import 'package:tiktok_practice/app/data/service/post_service.dart';
import 'package:video_player/video_player.dart';

class ViedeoControll with ChangeNotifier {
  VideoPlayerController? controller;
  int videoBefore = 0;
  int currentScreen = 0;
  List<Video> videoList = [];

  ViedeoControll._privateConstructor() {
    initializeVideoList();
  }

  static final ViedeoControll _instance = ViedeoControll._privateConstructor();

  factory ViedeoControll() {
    return _instance;
  }

  Future<List<Video>> initializeVideoList() async {
    videoList = await getVideoList();
    return videoList;
  }

  void changeVideo(int index) async {
    if (videoList[index].controller == null) {
      await videoList[index].loadController();
    }

    videoList[index].controller!.play();

    if (videoList[videoBefore].controller != null) {
      videoList[videoBefore].controller!.pause();
    }
    videoBefore = index;
  }

  void loadVideo(int index) async {
    if (index < videoList.length) {
      await videoList[index].loadController();
      videoList[index].controller?.play();
      notifyListeners();
    }
  }

  void setActualScreen(int index) {
    currentScreen = index;
    Logger().d(currentScreen);
    SystemChrome.setSystemUIOverlayStyle(
        index == 0 ? SystemUiOverlayStyle.light : SystemUiOverlayStyle.dark);
    notifyListeners();
  }
}
