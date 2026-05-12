@echo off
setlocal

set SCRIPT=src\dna_face_pipeline\3d_generation\FLAME_PyTorch\generate_from_parameters.py
set CSV=data\synthetic_dataset\synthetic_dataset_complete.csv
set OUT=website\images
set IMG=website\images

echo Regenerating GIFs with pink lips fix...

for %%S in (SYNTH_000016 SYNTH_000029 SYNTH_000052 SYNTH_000066 SYNTH_000270 SYNTH_000601 SYNTH_001977 SYNTH_002246) do (
    echo.
    echo === %%S ===
    python %SCRIPT% --sample-id %%S --dataset-csv %CSV% --output-dir %OUT% --image %IMG%\%%S.png
    if errorlevel 1 (
        echo FAILED: %%S
    ) else (
        copy /Y %OUT%\deca_%%S_lower\deca_%%S_lower_spin.gif %OUT%\%%S.gif 2>nul
        echo Done: %%S
    )
)

echo.
echo Copying spin GIFs to website/images/...
for %%S in (SYNTH_000016 SYNTH_000029 SYNTH_000052 SYNTH_000066 SYNTH_000270 SYNTH_000601 SYNTH_001977 SYNTH_002246) do (
    set SID_LOW=%%S
    call :tolower SID_LOW
    copy /Y %OUT%\deca_!SID_LOW!\deca_!SID_LOW!_spin.gif %OUT%\%%S.gif
)

echo All done!
exit /b 0

:tolower
for %%a in (A=a B=b C=c D=d E=e F=f G=g H=h I=i J=j K=k L=l M=m N=n O=o P=p Q=q R=r S=s T=t U=u V=v W=w X=x Y=y Z=z) do (
    set "str=!str:%%~a!"
)
exit /b 0
