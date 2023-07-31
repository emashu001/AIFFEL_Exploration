import 'package:flutter/material.dart';
import 'package:tiktok_practice/app/data/model/video.dart';
import 'package:video_player/video_player.dart';

import 'widgets/actionsToolBar.dart';
import 'widgets/videoDescription.dart';

class VideoCard extends StatefulWidget {
  final Video video;

  VideoCard(this.video);

  @override
  _VideoCardState createState() => _VideoCardState();
}

class _VideoCardState extends State<VideoCard> {
  VideoPlayerController? _controller;

  @override
  void initState() {
    super.initState();
    _initializeController();
  }

  @override
  void dispose() {
    _disposeController();
    super.dispose();
  }

  Future<void> _initializeController() async {
    _controller = VideoPlayerController.networkUrl(Uri.parse(widget.video.url));
    await _controller?.initialize();
    setState(() {});
  }

  Future<void> _disposeController() async {
    await _controller?.dispose();
  }

  void _togglePlayPause() {
    if (_controller?.value.isPlaying == true) {
      _controller?.pause();
    } else {
      _controller?.play();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        _controller != null && _controller!.value.isInitialized
            ? GestureDetector(
                onTap: _togglePlayPause,
                child: AspectRatio(
                  aspectRatio: _controller!.value.aspectRatio,
                  child: VideoPlayer(_controller!),
                ),
              )
            : Container(
                color: Colors.black,
                child: Center(
                  child: Text("Loading"),
                ),
              ),
        Column(
          mainAxisAlignment: MainAxisAlignment.end,
          children: <Widget>[
            Row(
              mainAxisSize: MainAxisSize.max,
              crossAxisAlignment: CrossAxisAlignment.end,
              children: <Widget>[
                VideoDescription(
                  widget.video.user,
                  widget.video.videoTitle,
                  widget.video.songName,
                ),
                ActionsToolbar(
                  widget.video.likes,
                  widget.video.comments,
                  widget.video.userPic,
                ),
              ],
            ),
            SizedBox(height: 20),
          ],
        ),
      ],
    );
  }
}

Widget videoCard(Video video) {
  return VideoCard(video);
}
