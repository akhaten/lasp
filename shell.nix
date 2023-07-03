{ pkgs ? import <nixpkgs> {}, ... }:

pkgs.mkShell {

    buildInputs = with pkgs; [

        python310
        python310Packages.numpy
        python310Packages.scipy
        python310Packages.matplotlib
        python310Packages.pandas
        python310Packages.tqdm

        # Package for Jupyter / To comment
        python310Packages.ipywidgets
        python310Packages.ipykernel
        python310Packages.ipympl
        python310Packages.ipython
        
        # TO DELETE
        python310Packages.scikitimage
        python310Packages.scikit-learn

   

        

    ];

}
