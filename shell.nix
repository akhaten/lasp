{ pkgs ? import <nixpkgs> {}, ... }:

pkgs.mkShell {

    buildInputs = with pkgs; [

        python310
        python310Packages.numpy
        python310Packages.scipy
        python310Packages.matplotlib
        python310Packages.pandas


        # Package for Jupyter / To comment
        python310Packages.ipywidgets
        python310Packages.ipykernel
        python310Packages.ipympl

        # TO DELETE
        python310Packages.scikitimage
        python310Packages.scikit-learn

   

        

    ];

}
