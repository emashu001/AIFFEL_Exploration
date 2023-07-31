import 'package:video_player/video_player.dart';

class Video {
  String id;
  String user;
  String userPic;
  String videoTitle;
  String songName;
  String likes;
  String comments;
  String url;

  VideoPlayerController? controller;

  Video({
    required this.id,
    required this.user,
    required this.userPic,
    required this.videoTitle,
    required this.songName,
    required this.likes,
    required this.comments,
    required this.url,
  });

  Video.fromJson(Map<dynamic, dynamic> json)
      : id = json['id'] ?? '',
        user = json['user'] ?? '',
        userPic = json['user_pic'] ?? '',
        videoTitle = json['video_title'] ?? '',
        songName = json['song_name'] ?? '',
        likes = json['likes'] ?? '',
        comments = json['comments'] ?? '',
        url = json['url'] ?? '';

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'user': user,
      'user_pic': userPic,
      'video_title': videoTitle,
      'song_name': songName,
      'likes': likes,
      'comments': comments,
      'url': url,
    };
  }

  Future<void> loadController() async {
    Uri urlLink = Uri.parse(url);
    controller = VideoPlayerController.networkUrl(urlLink);
    await controller?.initialize();
    controller?.setLooping(true);
  }

  Future<void> disposeController() async {
    await controller?.dispose();
  }

  @override
  String toString() {
    return 'Video(id: $id, user: $user, userPic: $userPic, videoTitle: $videoTitle, songName: $songName, likes: $likes, comments: $comments, url: $url)';
  }
}
