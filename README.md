# mz-forest-sound-track

Our project addresses the [City Forest Visitors](https://makezurich.ch/box/2/) challenge using sound measurements and machine learning techniques. We have creatively interpreted "Visitors" to also mean animals, initially focusing on birds.

Bird watching is a popular pastime activity, and several apps[1][2] exist for detecting and recording bird songs. We are trying to duplicate these functions while brainstorming how to stay within or close to the low-power parameters of **LoRaWAN**, because we believe there to be value in monitoring the sounds of the forest over long periods of time or in remote areas. The low cost approach facilitated by The Things Network ensures that this project could be reproduced by both scientists and amateurs alike.

Unlike smartphone apps, whose data depends on the user's schedule, the data produced by our sensor could potentially cover the entire day and night time spectrum. An important use case could also be monitoring the interaction between animals and humans. Our dataset could easily encompass bird and human sounds (footsteps, voices, ...) to study behavior patterns. 

#### What we tried (so far..)

- Did research into the idea, discovering numerous projects and even whole competitions to automatically detect and classify bird song. These led us to datasets, sample code, and pro tips.
- Looked into the possibility of doing audio classification on low-power chips (Atmega/Arduino) and discussed the idea with an expert in Arduino sound sensors (Baxter). Despite interesting libraries[1] the low recording quality and limited processing power and capacity would limit our options here severely. 
- We discussed the issue of power draw of the Raspberry Pi, to find out how we could activate the device on a timer (use a MOSFET) and prolong battery life (use the sun).
- Looked into the option of attaching a LoRaWAN antenna directly to Raspberry Pi, but decided to keep an Arduino as part of our hack with which we communicate via USB serial.
- Started putting together a training dataset with crowdsourced (Xeno-Canto) bird song samples from the Zürich region.
- Starting work on a training and classification script.

### Data sources

- [Zürich bird recordings](https://www.xeno-canto.org/explore?query=box%253A47.248%252C8.183%252C47.51%252C8.799+&dir=0&order=elev)

### References

Use cases, libraries, interesting datasets and example projects can currently be found in our Slack channel #mz-forest-sound-track

### Who are we

- [@afsoonica](https://github.com/afsoonica)
- [@loleg](https://github.com/loleg)
