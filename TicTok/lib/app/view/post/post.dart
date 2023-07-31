import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:tiktok_practice/app/data/service/video_controll.dart';
import 'package:tiktok_practice/app/view/post/scroll_post.dart';

import '../mypage/mypage.dart';
import '../uploard/upload.dart';

class PostScreen extends StatefulWidget {
  PostScreen({Key? key}) : super(key: key);

  @override
  _PostScreenState createState() => _PostScreenState();
}

class _PostScreenState extends State<PostScreen> {
  ViedeoControll vdController = ViedeoControll();
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor:
          vdController.currentScreen == 0 ? Colors.black : Colors.white,
      body: Stack(
        children: [
          PageView.builder(
            itemCount: 3,
            onPageChanged: (value) {
              print(value);
              if (value == 1)
                //TODO 시스템 크롬
                SystemChrome.setSystemUIOverlayStyle(SystemUiOverlayStyle.dark);
              else
                SystemChrome.setSystemUIOverlayStyle(
                    SystemUiOverlayStyle.light);
            },
            itemBuilder: (context, index) {
              if (index == 0) {
                return scrollPost();
              } else if (index == 1) {
                return ProfileScreen();
              } else {
                return UploadScreen();
              }
            },
          )
        ],
      ),
    );
  }
}
