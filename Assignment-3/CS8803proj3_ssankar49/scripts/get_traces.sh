TRACE_FILE=macsim_traces.tar.gz

GDRIVE_FILEID="1rpAgIMGJnrnXwDSiaM3S7hBysFoVhyO1"

if test -d ${MACSIM_TRACE_DIR}; then
  echo "> Traces found!  (${MACSIM_TRACE_DIR})";
else
  if [ ${MACSIM_TRACE_DIR} = "${MACSIM_DIR}/macsim_traces" ]; then
    # We're using local trace directory 
    echo "> Downloading traces...";
    gdown -O ${TRACE_FILE} ${GDRIVE_FILEID}

    if [ $? -eq 0 ]; then
      echo "> Download ${TRACE_FILE}: OK"
    else
      echo "> Download ${TRACE_FILE}: FAILED"
      exit 1
    fi

    # Extract traces
    tar -xzf ${TRACE_FILE};
    rc=$?
    rm -f ${TRACE_FILE};

    if [ ${rc} -eq 0 ]; then
      echo "> Extracting ${TRACE_FILE}: OK"
    else
      echo "> Extracting ${TRACE_FILE}: FAILED"
      exit 1
    fi
  
  else
    echo "ERROR: Trace directory not setup properly"
    exit 1
  fi
fi
