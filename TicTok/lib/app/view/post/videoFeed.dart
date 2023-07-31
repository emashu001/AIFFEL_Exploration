import 'package:flutter/material.dart';
import 'package:tiktok_practice/app/data/service/video_controll.dart';
import 'package:tiktok_practice/app/view/post/videoCard.dart';
import '../../data/model/video.dart';

Widget feedVideos() {
  ViedeoControll viedeoControl = ViedeoControll();

  return FutureBuilder<List<Video>>(
    future: viedeoControl.initializeVideoList(),
    builder: (context, snapshot) {
      if (snapshot.connectionState == ConnectionState.waiting) {
        return Center(
          child: SizedBox(
            width: 50,
            height: 50,
            child: CircularProgressIndicator(),
          ),
        );
      } else if (snapshot.hasError) {
        return Center(
          child: Text('Error loading videos'),
        );
      } else {
        return Stack(
          children: [
            PageView.builder(
              controller: PageController(
                initialPage: 0,
                viewportFraction: 1,
              ),
              itemCount: snapshot.data!.length,
              onPageChanged: (index) {
                index = index % snapshot.data!.length;
                viedeoControl.changeVideo(index);
              },
              scrollDirection: Axis.vertical,
              itemBuilder: (context, index) {
                index = index % snapshot.data!.length;
                return videoCard(snapshot.data![index]);
              },
            ),
            SafeArea(
              child: Container(
                padding: EdgeInsets.only(top: 20),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: <Widget>[
                    Text('팔로잉',
                        style: TextStyle(
                            fontSize: 17.0,
                            fontWeight: FontWeight.normal,
                            color: Colors.white70)),
                    SizedBox(
                      width: 7,
                    ),
                    Container(
                      color: Colors.white70,
                      height: 10,
                      width: 1.0,
                    ),
                    SizedBox(
                      width: 7,
                    ),
                    Text('❤️',
                        style: TextStyle(
                            fontSize: 17.0,
                            fontWeight: FontWeight.bold,
                            color: Colors.white))
                  ],
                ),
              ),
            ),
          ],
        );
      }
    },
  );
}
