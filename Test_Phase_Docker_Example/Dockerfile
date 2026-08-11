FROM anibali/pytorch:2.0.1-cuda11.8-ubuntu22.04

# Optional: set working directory
WORKDIR /app

# Set up time zone
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Copy requirements first for caching
COPY ./requirements.txt .

# Install system and Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy only necessary files for runtime
COPY ./infer_convexadam.py /app/
COPY ./MIR /app/MIR
RUN sudo chmod -R a+rw /app/MIR
# Install the MIR package (editable if you plan to modify)
WORKDIR /app/MIR
RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/input /app/output
RUN chmod -R 777 /app/output
# Switch back to /app as the working directory
WORKDIR /app

# ANTs affine + ConvexAdam-MIND (SVF) registration - one moving(EXP)/fixed(INSP) CT pair per
# run. The registration is optimization-based (no learned weights) and the container makes no
# assumption about filenames: the caller passes three paths - fixed (INSP) CT, moving (EXP)
# CT, output. It writes one original-resolution displacement field, composed from the ANTs
# affine and the ConvexAdam deformable fields.
#
#   docker run ... <image>  /app/input/FIXED_INSP.nii.gz  /app/input/MOVING_EXP.nii.gz \
#                           /app/output/DISP.nii.gz
ENTRYPOINT ["python3","-u","/app/infer_convexadam.py"]
