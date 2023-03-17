{ pkgs ? import <nixpkgs> {}, ... }:

pkgs.mkShell {

    buildInputs = with pkgs; [

        python310
        python310Packages.numpy
        python310Packages.scipy
        python310Packages.scikitimage
        python310Packages.matplotlib
        python310Packages.pandas


        # Package for Jupyter
        python310Packages.ipywidgets
        python310Packages.ipykernel
        python310Packages.ipympl

        

    ];

}
