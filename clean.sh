read -p "Are you sure to clean? (Y/N):" para
case $para in
    [yY])
        rm -rf ./vocab.json
        rm -rf ./summary
        rm -rf ./checkpoints
        rm -rf __pycache__
        ;;
    *)
        exit 1
        ;;
esac
