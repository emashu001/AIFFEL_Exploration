import 'dart:io';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:tiktok_practice/app/view/auth/login.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

import '../../view/post/post.dart';

Future<void> signInWithEmail(
    String email, String password, BuildContext context) async {
  try {
    UserCredential userCredential =
        await FirebaseAuth.instance.signInWithEmailAndPassword(
      email: email,
      password: password,
    );

    if (userCredential.user != null) {
      // 로그인 완료 후 홈 화면으로 이동
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (context) => PostScreen()),
      );
    }
  } catch (e) {
    print('Error signing in with email: $e');
    // TODO: Show error message to user
  }
}

Future<void> signUpWithEmailAndImage(
  String email,
  String password,
  String name,
  File imageFile,
  BuildContext context,
) async {
  try {
    // 회원가입
    UserCredential userCredential = await FirebaseAuth.instance
        .createUserWithEmailAndPassword(email: email, password: password);

    if (userCredential.user != null) {
      String userId = userCredential.user!.uid;

      // 프로필 이미지 업로드
      String? imageUrl = await uploadProfileImage(userId, imageFile);

      // Firestore에 사용자 정보 저장
      await FirebaseFirestore.instance.collection('users').doc(userId).set({
        'uuid': userId,
        'name': name,
        'image_url': imageUrl,
      });

      // 사용자 이름 업데이트
      await userCredential.user!.updateDisplayName(name);

      // 회원가입 완료 후 로그인 화면으로 돌아가기
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (context) => LoginScreen()),
      );
    }
  } catch (e) {
    print('Error signing up with email and image: $e');
    // TODO: Show error message to user
  }
}

Future<String?> uploadProfileImage(String userId, File imageFile) async {
  try {
    Reference storageRef = await FirebaseStorage.instance
        .ref()
        .child('profile_/images/$userId.jpg');
    UploadTask uploadTask = storageRef.putFile(imageFile);
    TaskSnapshot snapshot = await uploadTask;
    String downloardUrl = await snapshot.ref.getDownloadURL();
    return downloardUrl;
  } catch (e) {}
}

Future<String?> getCurrentUserName() async {
  User? user = FirebaseAuth.instance.currentUser;
  if (user != null) {
    await user.reload(); // 사용자 정보 갱신
    user = FirebaseAuth.instance.currentUser; // 갱신된 사용자 정보 가져오기
    return user!.displayName;
  }
  return null;
}

Future<String?> getCurrentUserUuid() async {
  User? user = FirebaseAuth.instance.currentUser;
  if (user != null) {
    await user.reload(); // 사용자 정보 갱신
    user = FirebaseAuth.instance.currentUser;
    return user!.uid;
  }
  return null;
}

Future<String?> getProfileImageUrl() async {
  User? user = FirebaseAuth.instance.currentUser;
  String uid = user!.uid;
  try {
    QuerySnapshot querySnapshot = await FirebaseFirestore.instance
        .collection('users')
        .where('uuid', isEqualTo: uid)
        .limit(1)
        .get();

    if (querySnapshot.docs.isEmpty) {
      print('No user found with uuid: $uid');
      return null;
    }

    DocumentSnapshot snapshot = querySnapshot.docs.first;

    if (snapshot.exists) {
      String? imageUrl = snapshot.get("image_url");
      return imageUrl;
    }
  } catch (e) {
    print('Error getting profile image URL: $e');
  }
  return null;
}
