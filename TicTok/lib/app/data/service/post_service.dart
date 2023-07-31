//TODO 비디오 리스트 받기
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:tiktok_practice/app/data/model/video.dart';

Future<List<Video>> getVideoList() async {
  var videos = await FirebaseFirestore.instance.collection("Videos").get();

  /*if (videos.docs.isEmpty) {
    videos = await FirebaseFirestore.instance.collection("Videos").get();
  }*/

  return videos.docs.map((doc) => Video.fromJson(doc.data())).toList();
}

Future<List<String>> getMyVideoList() async {
  User? user = FirebaseAuth.instance.currentUser;
  String uid = user!.uid;
  final userDocRef =
      FirebaseFirestore.instance.collection('users').doc(uid.toString());

  final userDocSnapshot = await userDocRef.get();
  if (userDocSnapshot.exists) {
    final myVideoList =
        userDocSnapshot.data()?['myVideoList'] as List<dynamic>? ?? [];
    return myVideoList.map((url) => url.toString()).toList();
  } else {
    return [];
  }
}
