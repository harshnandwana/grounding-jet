## How the questions were created

The published train, validation, and test rows largely use fixed question templates. COCO instance boxes and category labels yield normalized grounding boxes, box-choice questions with two distractor boxes, and left/right decisions for clearly separated present objects. Visual Genome v1.2 supplies selected color attributes and object relations for photos overlapping COCO. Objects, coordinates, answers, and available task counts vary by photo. The 6,823 more varied Luna candidate rows remain outside all published splits pending human audit. These data do not validate unfamiliar wording, counting, absence, OCR, or false-premise questions. See the [builder code and selection details](https://github.com/harshnandwana/grounding-jet/blob/main/docs/DATA.md).

One held-out photo yields 61 records: 8 grounding, 8 box choice, 38 spatial, 4 color, and 3 relation. The [GitHub figure](https://github.com/harshnandwana/grounding-jet/blob/main/results/one_image_dataset.png) shows the source photo with one published question and answer from each task; the [row IDs](https://github.com/harshnandwana/grounding-jet/blob/main/results/one_image_dataset.json) make it reproducible. The figure is hosted on GitHub, **not uploaded as a photo file to this dataset repository**.

![One source image yielding five kinds of dataset question](https://raw.githubusercontent.com/harshnandwana/grounding-jet/main/results/one_image_dataset.png)

