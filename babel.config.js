module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    // VisionCamera v4 frame processors use worklets-core. Skia needs Reanimated,
    // whose plugin (react-native-worklets) must stay last.
    plugins: ['react-native-worklets-core/plugin', 'react-native-reanimated/plugin'],
  };
};
