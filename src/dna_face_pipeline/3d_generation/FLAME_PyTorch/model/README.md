# FLAME Model Assets

Put the local FLAME model files in this folder.

Required filenames:

- `generic_model.pkl`
- `flame_static_embedding.pkl`
- `flame_dynamic_embedding.npy`

These files are not included in this repository because the FLAME model is
licensed separately. Download the FLAME model from:

https://flame.is.tue.mpg.de/

The landmark embedding files come from the RingNet `flame_model` assets:

https://github.com/soubhiksanyal/RingNet/tree/master/flame_model

After placing the files here, the generator can be run from the repository root:

```powershell
python src\dna_face_pipeline\3d_generation\generate_from_parameters.py --sample-id SYNTH_000052
```
