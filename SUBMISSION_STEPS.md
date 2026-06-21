# How to Submit This Project

The submission package is ready. Use the folder:

```text
assignment_submission_medical_image_agent_final/
```

or the zip file:

```text
assignment_submission_medical_image_agent_final.zip
```

## Web Upload Method

1. Open the assignment repository:
   <https://github.com/Yuplx-HU/ai-agent-assignment>

2. Click **Fork** to create your own copy.

3. Open your forked repository.

4. Upload the contents of `assignment_submission_medical_image_agent_final/` into your fork.

5. Commit the uploaded files.

6. Click **Contribute** → **Open pull request**.

7. Use the text in `PR_DESCRIPTION.md` as the pull request description.

## Git Command Method

Replace `<your-github-name>` with your GitHub username:

```bash
git clone https://github.com/<your-github-name>/ai-agent-assignment.git
cd ai-agent-assignment
```

Copy the contents of `assignment_submission_medical_image_agent_final/` into the cloned repository, then run:

```bash
git add .
git commit -m "Submit medical image segmentation AI agent project"
git push origin main
```

Then open a Pull Request from your fork to:

```text
Yuplx-HU/ai-agent-assignment:main
```

## Important Safety Notes

Do not commit:

- `.env`
- API keys
- `model_weights/`
- `code/npc_dataset_nii/`
- `*.nii.gz`
- `*.npy`
- `*.pth`

These are already excluded in `.gitignore`.
