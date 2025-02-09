# ImplementingTransformers

Welcome to the Implementing Transformers Practical Course.

## Session Times

The sessions are designated for questions concerning implementations and the current practical (the course is a practical course there will be no "lectures").

Monday: 16:30-18:00 (Gebäude 25.22 / U1.52)

Wednesday: 16:30-18:00 (Gebäude 25.12 / 01.51)

## Contact details

Join the Rocket Chat group: https://rocketchat.hhu.de/channel/Implementing-Transformer-Models

Email: cvanniekerk@hhu.de / linh@hhu.de

## Schedule

| Week | Dates         | Practical                                              | Evaluation                                     |
|------|---------------|--------------------------------------------------------|------------------------------------------------|
| 1    | 7-11.10.2024  | Practical 1: Getting Started and Introduction to Transformers and Attention |                                                |
| 2    | 14-18.10.2024 | Practical 2: Introduction to Unit Tests and Masked Attention |                                                |
| 3    | 21-25.10.2024 | Practical 3: Tokenization                               | Test 1 |
| 4    | 28-31.10.2024 | Practical 4: Data Preparation and Embedding Layers      |                                                |
| 5    | 4-8.11.2024   | Practical 5: Multi-Head Attention and Transformer Encoder                |                                                |
| 6    | 11-15.11.2024 | Practical 6: Transformer Encoder and Decoder Layers     |             |
| 7    | 18-22.11.2024 | Practical 6: Transformer Encoder and Decoder Layers     | Test 2   |
| 8    | 25-29.11.2024 | Practical 7: Complete Transformer Model                 |                                                |
| 9    | 2-6.12.2024   | Practical 8: Training and Learning rate Schedules       |                                                |
| 10   | 9-13.12.2024  | Practical 9: Training the model                        | Test 3      |
| 11   | 16-20.12.2024 | Practical 9: Training the model                       |                                                |
| 12   | 6-11.01.2025  | Practical 10: Autoregressive Generation and Evaluation |                                                |
| 13   | 13-17.01.2025 | Practical 11: GPU Training (Colab)                     |                                                |
| 14   | 20-24.01.2025 |                                                        | Test 4          |

## Evaluation

The evaluation for this course will be based on two components: class tests and a final project.

### Class Tests

Throughout the semester, there will be a total of four class tests. These tests are designed to assess your understanding of the key concepts and practical exercises covered in the course. Each test will focus on specific topics covered up to that point in the course.

**Eligibility for Final Project:** In order to be eligible to submit the final project, students must achieve an average of 50% across the four tests.

### Final Project

The final project consists of both a written report and an oral presentation. This project is your opportunity to showcase the depth of your understanding and ability to apply the concepts learned throughout the course.

Report Guidelines:

**Word Limit:** The report should not exceed 2500 words.

**Page Limit:** The report must be a maximum of 8 pages.

**Content:**
Your report should be comprehensive yet concise, ensuring that it is self-contained and covers all necessary topics.
A brief introduction is recommended, providing an overview of the transformer model and laying the groundwork for the report.
The report must include experimental results and responses to the questions from the practical sheets. When incorporating responses, ensure they are seamlessly integrated into the narrative rather than being presented in a Q&A format.
Conclude the report with a summary that encapsulates the main findings and insights gained from the practicals.

Structure: While your report may follow the structure of a standard research paper, it is important that your content and perspective are original. It must not simply replicate the "Attention Is All You Need" paper, but instead should provide your unique insights and understanding based on the course work.

Word Count: Please include the word count at the end of your report. Note that equations are exempt from the word count.

Code Submission: Your code should be submitted along with your report. You may either:
Include a link to your Git repository (make sure to grant me access), or
Submit a compressed copy of your code files.

Note: To achieve a distinction final grade, your report should not only illustrate an understanding of the course material but also demonstrate originality and creativity. To do so it is required to include a extra small-scale experiments and analysis of some variation to the transformer model, the task, the training setup etc. Find below a list of potential project topics to inspire you.

**Potential Project Topics:**

- Linear / Window-size Attention
  - Compare the computational complexity with that of the standard attention mechanism.
  - What are the positives and drawbacks of this approach?
  - Small scale (synthetic data) experiments to demonstrate the effectiveness of such an approach.
- Decoder-only and or Encoder-only Transformer
  - What are the advantages and disadvantages of training a decoder-only or encoder-only transformer?
  - Why are modern LLMs decoder-only?
  - Small scale (synthetic data) experiments to demonstrate the effectiveness of such an approach.
- Fine-tuning vs Soft-Prompting
  - Compare the performance of fine-tuning vs soft-prompting on a specific task.
  - How does LoRA fine tuning work?
  - What are the advantages and disadvantages of each approach?
  - Small scale (synthetic data) experiments to demonstrate the effectiveness of this approach (Test using the sort increasing, decreasing +1 and +2 synthetic task on a sequence of numbers).
- Trainable vs Sinusoidal Positional Encodings
  - Compare the performance of trainable vs sinusoidal positional encodings.
  - What are the advantages and disadvantages of each approach?
  - Small scale (synthetic data) experiments to demonstrate the effectiveness of this approach.
  - Can sinusoidal positional encodings really generalise to unseen lengths?
- GPU Training
  - Compare the training time of the transformer model on a CPU vs GPU.
  - What are the advantages and disadvantages of training on a GPU?
  - What kind of tasks can a GPU perform.
  - How can one estimate how much memory is required for a specific task?

**Presentation Guidelines:**
Prepare a 10-minute presentation highlighting the most important aspects of your report.
The presentation should focus on key insights, challenges, and outcomes from your project.

**Submission Deadlines.**
- Report: 29.01.2025
- Presentation dates: 3-5.02.2025

**Submission:**
- Remember to send your code together with your report. You can either add a link to your git repo (remember to grant me access) or a copy of your code (compressed).
- Please send your final report to: cvanniekerk@hhu.de with the subject "Implementing Transformer Model Final Handin"

Good luck, and I look forward to seeing the results of your hard work!
