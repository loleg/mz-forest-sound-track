Our project addresses the [City Forest Visitors](https://makezurich.ch/box/2/) challenge using sound measurements and machine learning techniques. We have creatively interpreted "Visitors" to also mean animals, initially focusing on birds.

#### Why are we doing this?

Bird watching is a popular pastime activity, and several apps[1][2] exist for detecting and recording bird songs. We are trying to duplicate these functions while brainstorming how to stay within or close to the low-power parameters of **LoRaWAN**, because we believe there to be value in monitoring the sounds of the forest over long periods of time or in remote areas. The low cost approach facilitated by The Things Network ensures that this project could be reproduced by both scientists and amateurs alike.

Unlike smartphone apps, whose data depends on the user's schedule, the data produced by our sensor could potentially cover the entire day and night time spectrum. An important use case could also be monitoring the interaction between animals and humans. Our dataset could easily encompass bird and human sounds (footsteps, voices, ...) to study behavior patterns.

#### What we tried (so far..)

- Did research into the idea, discovering numerous projects and even whole competitions to automatically detect and classify bird song. These led us to datasets, sample code, and pro tips.
- Looked into the possibility of doing audio classification on low-power chips (Atmega/Arduino) and discussed the idea with an expert in Arduino sound sensors (Baxter). Despite interesting libraries ([walrus](https://github.com/walrus/walrus), [Neurona](http://www.moretticb.com/Neurona/)) and research ([Evolutionary Bits'n'Spikes](https://infoscience.epfl.ch/record/63939) from EPFL!) the low recording quality and limited processing power and capacity would limit our options here severely.
- We discussed the issue of power draw of the Raspberry Pi, to find out how we could activate the device on a timer (use a MOSFET) and prolong battery life (use the sun).
- Looked into the option of attaching a LoRaWAN antenna directly to Raspberry Pi, but decided to keep an Arduino as part of our hack with which we communicate via USB serial.
- Started putting together a training dataset with crowdsourced (Xeno-Canto) bird song samples from the Zürich region.
- Starting work on a training and classification script.
- Started an example [forest map](https://map.geo.admin.ch/?lang=en&topic=ech&bgLayer=ch.swisstopo.pixelkarte-farbe&layers=ch.swisstopo.zeitreihen,ch.bfs.gebaeude_wohnungs_register,ch.bav.haltestellen-oev,ch.swisstopo.swisstlm3d-wanderwege,ch.swisstopo.vec200-landcover-wald,ch.bafu.bundesinventare-vogelreservate,KML%7C%7Chttps:%2F%2Fpublic.geo.admin.ch%2FnX-OP2f_RbukVK7KrgjDDA&layers_visibility=false,false,false,false,true,true,true&layers_timestamp=18641231,,,,,,&layers_opacity=1,1,1,1,0.75,0.75,1&E=2678722.79&N=1245542.59&zoom=4.492539968390444)

### Data sources

- [Zürich bird recordings](https://www.xeno-canto.org/explore?query=box%253A47.248%252C8.183%252C47.51%252C8.799+&dir=0&order=elev)
- [UrbanSound8K dataset](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html)
- [Forested areas of VECTOR200](https://www.geocat.ch/geonetwork/srv/eng/catalog.search#/metadata/de52b509-87c7-4f7e-be0b-ff3c557d2949)

### References

- [librosa](https://librosa.github.io/): audio processing in Python
- [TensorFlow for Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/)
- [Compiling TensorFlow on RasPi](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile#raspberry-pi) vs [TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/)
- TensorFlow [implementation of SoundNet](https://github.com/eborboihuc/SoundNet-tensorflow) and [Audio Recognition tutorial](https://www.tensorflow.org/tutorials/audio_recognition)
- Interfacing Arduino with Raspi: [Hackster](https://www.hackster.io/sankarCheppali/interfacing-arduino-with-raspberry-pi-6d9870), [Instructables](http://www.instructables.com/id/Raspberry-Pi-Arduino-Serial-Communication/)

More use cases, libraries, interesting datasets and example projects can currently be found in our Slack channel #mz-forest-sound-track

### Who are we

- [@afsoonica](https://github.com/afsoonica)
- [@loleg](https://github.com/loleg)
