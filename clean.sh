read -p "Are you sure to clean? (Y/N):" para
case $para in
    [yY])
        # rm -rf ./vocab.json
        rm -rf ./summary
        rm -rf ./checkpoints
        rm -rf __pycache__
        rm -rf .datacache
        rm -rf ./ext/__pycache__
        rm -rf ./ext/.pytest_cache
        rm -rf ./ext/build
        rm -rf ./ext/dist
        rm -rf ./ext/*.egg-info
        ;;
    *)
        exit 1
        ;;
esac
