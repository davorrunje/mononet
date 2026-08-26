# shellcheck shell=bash
# Git-aware prompt for the mononet devcontainer.
#
# Sourced from ~/.bashrc and ~/.zshrc by install-shell-prompt.sh. Replaces the
# stock "user@host:cwd$" prompt with "cwd (branch)$" -- the user and host are
# noise inside a container, the branch is not.
#
# A virtualenv prefix (e.g. "(mononet) ") is still prepended by the activate
# script via VIRTUAL_ENV_PROMPT / PS1 rewriting, so it survives this.

if [ -n "${BASH_VERSION:-}" ]; then

    # Debian ships __git_ps1 in git-sh-prompt, but the container has no
    # bash-completion loader to source it, so do it ourselves.
    if ! declare -F __git_ps1 >/dev/null 2>&1; then
        for _mononet_gitprompt in \
            /usr/lib/git-core/git-sh-prompt \
            /usr/share/git-core/contrib/completion/git-prompt.sh \
            /etc/bash_completion.d/git-prompt
        do
            # shellcheck disable=SC1090
            [ -r "$_mononet_gitprompt" ] && . "$_mononet_gitprompt" && break
        done
        unset _mononet_gitprompt
    fi

    if ! declare -F __git_ps1 >/dev/null 2>&1; then
        # Fallback: no git-sh-prompt available in this image.
        __git_ps1() {
            local branch
            branch=$(git symbolic-ref --short -q HEAD 2>/dev/null) ||
                branch=$(git rev-parse --short HEAD 2>/dev/null) || return 0
            printf "${1:- (%s)}" "$branch"
        }
    fi

    GIT_PS1_SHOWDIRTYSTATE=1      # * unstaged, + staged
    GIT_PS1_SHOWSTASHSTATE=1      # $ stashed
    GIT_PS1_SHOWUNTRACKEDFILES=1  # % untracked
    GIT_PS1_SHOWUPSTREAM=auto     # </>/= vs upstream
    export GIT_PS1_SHOWDIRTYSTATE GIT_PS1_SHOWSTASHSTATE \
        GIT_PS1_SHOWUNTRACKEDFILES GIT_PS1_SHOWUPSTREAM

    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;34m\]\w\[\033[00m\]\[\033[01;33m\]$(__git_ps1 " (%s)")\[\033[00m\]\$ '

    case "$TERM" in
        xterm* | rxvt* | screen* | tmux*)
            PS1="\[\e]0;\w\a\]$PS1"
            ;;
    esac

elif [ -n "${ZSH_VERSION:-}" ]; then

    autoload -Uz vcs_info
    zstyle ':vcs_info:*' enable git
    zstyle ':vcs_info:git:*' check-for-changes true
    zstyle ':vcs_info:git:*' unstagedstr '*'
    zstyle ':vcs_info:git:*' stagedstr '+'
    zstyle ':vcs_info:git:*' formats       ' (%b%u%c)'
    zstyle ':vcs_info:git:*' actionformats ' (%b|%a%u%c)'

    _mononet_precmd() { vcs_info; }
    autoload -Uz add-zsh-hook
    add-zsh-hook precmd _mononet_precmd

    setopt PROMPT_SUBST
    PROMPT='%F{blue}%~%f%F{yellow}${vcs_info_msg_0_}%f%# '

fi
