import 'dart:io';

import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter/material.dart';
import 'package:tiktok_practice/app/data/service/auth_service.dart';
import 'package:tiktok_practice/app/data/service/post_service.dart';
import 'package:tiktok_practice/app/util/video_thumbnail.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({Key? key}) : super(key: key);

  @override
  _ProfileScreenState createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  late String currentUserName;
  late String currentUserUuid;
  late String profileImageUrl;
  late List<String> videoUrls;

  @override
  void initState() {
    super.initState();
    initializeUserDetails();
  }

  Future<void> initializeUserDetails() async {
    currentUserName = (await getCurrentUserName())!;
    currentUserUuid = (await getCurrentUserUuid())!;
    profileImageUrl = getProfile().toString();
    videoUrls = await getMyVideoList();
    setState(() {}); // to refresh the widget after data is fetched
  }

  Future<String?> getProfile() async {
    try {
      String? currentUserUuid = await getCurrentUserUuid();
      if (currentUserUuid == null) {
        print('Error getting current user UUID');
        return null;
      }
      String? profileImageUrl = await getProfileImageUrl();
      return profileImageUrl;
    } catch (e) {
      print('Error getting profile: $e');
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Container(
        color: Colors.white,
        child: Column(
          children: [
            Container(
              decoration: const BoxDecoration(
                  border: Border(bottom: BorderSide(color: Colors.black12))),
              padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 20),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Icon(Icons.person_add_alt_1_outlined),
                  FutureBuilder<String?>(
                    future: getCurrentUserName(),
                    builder: (context, snapshot) {
                      if (snapshot.connectionState == ConnectionState.waiting) {
                        return const CircularProgressIndicator();
                      } else if (snapshot.hasError) {
                        return Text('Error: ${snapshot.error}');
                      } else {
                        return Text(
                          snapshot.data ?? '',
                          style: const TextStyle(
                              fontSize: 15, fontWeight: FontWeight.bold),
                        );
                      }
                    },
                  ),
                  const Icon(Icons.more_horiz)
                ],
              ),
            ),
            SingleChildScrollView(
              child: Column(
                children: [
                  const SizedBox(height: 15),
                  FutureBuilder<String?>(
                    future: getProfileImageUrl(),
                    builder: (context, snapshot) {
                      if (snapshot.connectionState == ConnectionState.waiting) {
                        return const SizedBox(
                          width: 50,
                          height: 50,
                          child: CircularProgressIndicator(),
                        );
                      } else if (snapshot.hasError) {
                        return const Icon(Icons.error);
                      } else {
                        return ClipOval(
                          child: CachedNetworkImage(
                            fit: BoxFit.cover,
                            imageUrl: snapshot.data ?? '',
                            height: 100.0,
                            width: 100.0,
                            placeholder: (context, url) => const SizedBox(
                              width: 50,
                              height: 50,
                              child: CircularProgressIndicator(),
                            ),
                            errorWidget: (context, url, error) =>
                                const Icon(Icons.error),
                          ),
                        );
                      }
                    },
                  ),
                  const SizedBox(height: 10),
                  FutureBuilder<String?>(
                    future: getCurrentUserName(),
                    builder: (context, snapshot) {
                      if (snapshot.connectionState == ConnectionState.waiting) {
                        return const CircularProgressIndicator();
                      } else if (snapshot.hasError) {
                        return Text('Error: ${snapshot.error}');
                      } else {
                        return Text(
                          '@${snapshot.data ?? ''}',
                          style: const TextStyle(
                              fontSize: 15, fontWeight: FontWeight.bold),
                        );
                      }
                    },
                  ),
                  const SizedBox(height: 20),
                  const Divider(),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      Column(
                        mainAxisAlignment: MainAxisAlignment.end,
                        children: [
                          const Icon(Icons.menu),
                          const SizedBox(
                            height: 7,
                          ),
                          Container(
                            color: Colors.black,
                            height: 2,
                            width: 55,
                          )
                        ],
                      ),
                      Column(
                        mainAxisAlignment: MainAxisAlignment.end,
                        children: [
                          const Icon(
                            Icons.favorite_border,
                            color: Colors.black26,
                          ),
                          const SizedBox(
                            height: 7,
                          ),
                          Container(
                            color: Colors.transparent,
                            height: 2,
                            width: 55,
                          )
                        ],
                      ),
                      Column(
                        mainAxisAlignment: MainAxisAlignment.end,
                        children: [
                          const Icon(
                            Icons.lock_outline,
                            color: Colors.black26,
                          ),
                          const SizedBox(
                            height: 7,
                          ),
                          Container(
                            color: Colors.transparent,
                            height: 2,
                            width: 55,
                          )
                        ],
                      ),
                    ],
                  ),
                  FutureBuilder<List<String>>(
                    future: getMyVideoList(),
                    builder: (BuildContext context,
                        AsyncSnapshot<List<String>> snapshot) {
                      if (snapshot.connectionState == ConnectionState.waiting) {
                        return const CircularProgressIndicator();
                      } else if (snapshot.hasError) {
                        return Text('Error: ${snapshot.error}');
                      } else {
                        if (snapshot.data!.isEmpty) {
                          return const Center(child: Text("표시할 비디오가 없습니다"));
                        } else {
                          return GridView.builder(
                            shrinkWrap: true,
                            physics: const NeverScrollableScrollPhysics(),
                            gridDelegate:
                                const SliverGridDelegateWithFixedCrossAxisCount(
                              crossAxisCount: 4, // for 3 items in a row
                              crossAxisSpacing: 4.0,
                              mainAxisSpacing: 4.0,
                            ),
                            itemCount: snapshot.data!.length,
                            itemBuilder: (BuildContext context, int index) {
                              return FutureBuilder<String>(
                                future: createThumbnailFromVideoUrl(
                                    snapshot.data![index]),
                                builder: (context, snapshot) {
                                  if (snapshot.connectionState ==
                                      ConnectionState.waiting) {
                                    return const CircularProgressIndicator();
                                  } else if (snapshot.hasError) {
                                    return Text('Error: ${snapshot.error}');
                                  } else {
                                    return Image.file(File(snapshot.data!));
                                  }
                                },
                              );
                            },
                          );
                        }
                      }
                    },
                  )
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
